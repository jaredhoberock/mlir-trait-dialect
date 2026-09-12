// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "DemandLedger.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <cstdint>
#include <string>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/xxhash.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>

#include <TraitTypeInterfaces.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <TraitTypes.cpp.inc>

namespace mlir::trait {

AttrTypeReplacer makeEndpointSealedReplacer() {
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [](TypeEqualityAttr eq)
          -> std::optional<std::pair<Attribute, WalkResult>> {
        return std::make_pair(Attribute(eq), WalkResult::skip());
      });
  return replacer;
}

AttrTypeReplacer makeGroundProjectionReplacer(
    std::function<std::optional<Type>(ProjectionType)> hop) {
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement(
      [hop = std::move(hop)](Type t) -> std::optional<Type> {
        auto projection = dyn_cast<ProjectionType>(t);
        if (!projection || isPolymorphicType(projection))
          return std::nullopt;
        return hop(projection);
      });
  return replacer;
}

void TraitDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <TraitTypes.cpp.inc>
  >();
}

std::string hashToSuffix(StringRef input) {
  uint64_t hash = llvm::xxHash64(input);
  std::string result;
  llvm::raw_string_ostream out(result);
  out << llvm::format("_h%016" PRIx64, hash);
  out.flush();
  return result;
}

std::string generateMangledNameSuffixFor(TypeRange typeArgs) {
  if (typeArgs.empty()) return "";

  std::string full;
  llvm::raw_string_ostream os(full);
  for (Type ty : typeArgs)
    os << "_" << ty;
  os.flush();

  return hashToSuffix(full);
}

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const DenseMap<Type,Type> &subst,
    ArrayRef<GenericTypeInterface> typeParams) {
  SmallVector<Type> concreteTypes;
  for (auto ty : typeParams)
    concreteTypes.push_back(applySubstitutionOnce(subst, ty));
  return generateMangledNameSuffixFor(concreteTypes);
}

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams) {
  SmallVector<Type> concreteTypes;
  for (auto ty : typeParams)
    concreteTypes.push_back(subst.apply(ty));
  return generateMangledNameSuffixFor(concreteTypes);
}


//===----------------------------------------------------------------------===//
// Ground projection resolution
//===----------------------------------------------------------------------===//

namespace {

/// The rewrite budget the fixed-point driver spends before declaring a
/// projection resolution nonterminating. One projection resolution exposes at
/// most one more, so a chain that has not settled within this many passes is
/// cyclic or oscillating.
constexpr unsigned kProjectionFixedPointMaxIterations = 64;

/// Reports a projection whose resolution does not terminate as a diagnostic at
/// the demand it arose under, and returns without ending the process.
///
/// The rewrite this reports has no normal form. The infallible normalizer hands
/// the driver's partial back spelled as written, so every spelling comparison
/// downstream treats the projection as unresolved -- a decline in the safe
/// direction. A caller that must refuse rather than decline threads the failure
/// through `tryNormalizeProjectionsToFixedPoint` instead of this reporter; this
/// entry only surfaces the diagnostic for the infallible normalizer's users.
///
/// The enclosing demand names where the projection was asked about; outside a
/// stage span there is no enclosing demand and the module location is all there
/// is to name. No in-tree program reaches this, and the checks in front of it
/// are why: an impl whose own associated-type binding projects back through
/// itself resolves to a spelling equal to the demand, so the lookup makes no
/// progress and the leftover walk reports the projection as unresolved; a
/// binding cycle across two impls either grows the type until the bounded
/// substitution driver refuses it, or oscillates without growing and is caught
/// by the rewrite budget of the driver that keeps re-deriving it.
void reportUnnormalizableProjection(Type ty, unsigned iterations,
                                    ModuleOp module) {
  Location anchor = currentDemandAnchor().value_or(module.getLoc());
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "projection normalization did not converge within " << iterations
         << " iterations for type " << ty;
  emitError(anchor) << message;
}

} // namespace

LogicalResult tryNormalizeProjectionsToFixedPoint(
    Type ty, llvm::function_ref<Type(Type)> step, Type &out) {
  Type previous;
  for (unsigned i = 0;
       i != kProjectionFixedPointMaxIterations && ty != previous; ++i) {
    previous = ty;
    ty = step(ty);
  }
  // `out` carries what the loop reached either way: the fixed point on success,
  // the still-changing partial normal form on failure. A caller reporting the
  // nonconvergence reads the partial to name the type that would not settle.
  out = ty;
  return success(ty == previous);
}

Type normalizeProjectionsToFixedPoint(Type ty, ModuleOp module,
                                      llvm::function_ref<Type(Type)> step) {
  // Reaching the iteration cap while the type is still changing means the
  // rewrite has no fixed point (a cyclic or oscillating resolution). What the
  // loop reached is a partial normal form; this infallible entry surfaces the
  // nonconvergence as a diagnostic and hands the partial back spelled as
  // written. Every caller here compares that spelling against another or stamps
  // it, and an unresolved projection declines in the safe direction at each --
  // a spelling mismatch, never a silent accept. A caller that must refuse the
  // cycle rather than decline threads the failure through
  // `tryNormalizeProjectionsToFixedPoint` instead.
  Type out;
  if (failed(tryNormalizeProjectionsToFixedPoint(ty, step, out)))
    reportUnnormalizableProjection(out, kProjectionFixedPointMaxIterations,
                                   module);
  return out;
}

// Shared body of both projection-resolution entry points. `converged` reports
// whether the fixed-point driver reached a normal form: on false, `ty` carries
// the driver's partial (the still-unresolved projection spelled as written),
// and the two public overloads decide how to surface the nonconvergence -- the
// infallible one declines on the partial after a diagnostic, the fallible one
// refuses.
static Type resolveProjectionsByLookupCore(Type ty, ModuleOp module,
                                           DemandOrigin origin, LookupScope scope,
                                           bool &converged) {
  converged = true;
  if (!module)
    return ty;

  // Candidate impls per trait application, memoized for this resolution. The
  // lookup mutates no impls, so the memo stays valid across the fixed-point
  // iterations below, and it is scoped to this call so nothing outside observes
  // it -- repeated projections over the same application skip the module scan.
  DenseMap<TraitApplicationAttr, SmallVector<ImplOp>> candidateCache;

  // The context a candidate's header is read through here: this lookup itself,
  // so a header spelling a projection (`impl<T> Index<T::Shape, T::Element> for
  // T`) reproduces a demand spelling the resolution.
  GroundProjectionLookup byGroundLookup(module, origin);

  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
    // Impl enumeration below matches each candidate's header, which normalizes,
    // which re-enters this callback. The guard makes that re-entry visible, so
    // a demand raised about a candidate is told apart from the demand this call
    // was asked about.
    LookupProbeScope probe;

    const bool polymorphic = isPolymorphicType(proj);

    auto declineWith = [&](LookupMissReason reason) {
      // A demand is a question put to the impl engine about one type, and only a
      // ground projection asks one: a spelling that still carries variables
      // stands for as many types as its variables have instances, so no engine
      // owes it an answer and the ledger has nothing to record. The scope below
      // reads such a spelling to compare it, never to serve it.
      if (polymorphic)
        return std::optional<Type>(std::nullopt);
      recordLookupMiss(Type(proj), reason, origin, probe.getEnclosingDepth());
      return std::optional<Type>(std::nullopt);
    };

    // A projection whose arguments still carry variables resolves only under the
    // determined scope, and then only if its own spelling picks the impl (the
    // one-way match below).
    if (polymorphic && scope == LookupScope::Ground)
      return std::nullopt;

    ClaimType claim = proj.asClaim();
    TraitApplicationAttr app = claim.getTraitApplication();

    // Read-only selection: resolve only when exactly one existing impl binds
    // this application. Two or more matches, and impl generation, are left to
    // the resolver. The single match may be conditional (a nonempty assumptions
    // list): selecting it is mechanical name resolution, not premise evaluation,
    // and a legal program has already discharged this ground projection's head
    // claim -- the premise the conditional impl carries.
    auto it = candidateCache.find(app);
    if (it == candidateCache.end()) {
      auto trait = app.getTrait(module, nullptr);
      if (failed(trait))
        return declineWith(LookupMissReason::TraitSymbolNotFound);
      it = candidateCache
               .insert({app, trait->getCandidateImplsFor(claim, byGroundLookup)})
               .first;
    }
    const SmallVector<ImplOp> &candidates = it->second;
    if (candidates.size() != 1)
      return declineWith(candidates.empty()
                             ? LookupMissReason::NoCandidateImpl
                             : LookupMissReason::MultipleCandidateImpls);
    ImplOp impl = candidates.front();

    SmallVector<Type> assocTypeArgs(proj.getAssocTypeArgs());
    auto binding = impl.specializeAssociatedTypeBinding(
        proj.getAssocName().getValue(), assocTypeArgs);
    if (failed(binding))
      return declineWith(LookupMissReason::AssociatedTypeBindingFailed);
    // Nothing the projection spells is narrowed to fit the impl: the impl's own
    // parameters take the arguments standing opposite them and the header
    // rebuilt at those must be the projection's application. So an impl the
    // projection could only reach by narrowing one of its variables is refused
    // here, and no separate one-way test stands over this one.
    auto subst = impl.buildSubstitutionForSelfClaim(claim, byGroundLookup,
                                                    /*errFn=*/nullptr);
    if (failed(subst))
      return declineWith(LookupMissReason::SelfClaimSubstitutionFailed);

    return instantiate(*binding, *subst);
  });

  // A resolved binding may itself expose a ground projection, so run to a
  // fixed point. A chain that never grounds leaves `ty` at the driver's partial
  // normal form and reports nonconvergence to the caller.
  converged = succeeded(tryNormalizeProjectionsToFixedPoint(
      ty, [&](Type t) { return replacer.replace(t); }, ty));
  return ty;
}

Type resolveProjectionsByLookup(Type ty, ModuleOp module, DemandOrigin origin,
                                LookupScope scope) {
  bool converged;
  Type out = resolveProjectionsByLookupCore(ty, module, origin, scope,
                                            converged);
  // The infallible entry cannot refuse. A projection that will not ground stays
  // spelled as written in `out`, so every spelling comparison downstream
  // declines on it in the safe direction; the reporter surfaces the diagnostic.
  if (!converged)
    reportUnnormalizableProjection(out, kProjectionFixedPointMaxIterations,
                                   module);
  return out;
}

FailureOr<Type> resolveProjectionsByLookup(
    Type ty, ModuleOp module, DemandOrigin origin, LookupScope scope,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  bool converged;
  Type out = resolveProjectionsByLookupCore(ty, module, origin, scope,
                                            converged);
  // The fallible entry refuses a projection that will not ground so a verifier
  // reached from untrusted IR fails cleanly rather than admitting the cycle.
  if (!converged) {
    if (emitError)
      emitError() << "projection normalization did not converge within "
                  << kProjectionFixedPointMaxIterations
                  << " iterations for type " << out;
    return failure();
  }
  return out;
}

//===----------------------------------------------------------------------===//
// TypeEquivalence
//===----------------------------------------------------------------------===//

unsigned TypeEquivalence::intern(Type t) {
  auto it = ids.find(t);
  if (it != ids.end())
    return it->second;
  unsigned id = terms.size();
  ids[t] = id;
  terms.push_back(t);
  parent.push_back(id);
  orderKeys.emplace_back();
  return id;
}

unsigned TypeEquivalence::findCanonical(unsigned id) {
  while (parent[id] != id) {
    parent[id] = parent[parent[id]];
    id = parent[id];
  }
  return id;
}

void TypeEquivalence::unite(unsigned a, unsigned b) {
  a = findCanonical(a);
  b = findCanonical(b);
  if (a == b)
    return;
  // The joined class keeps the lesser of the two canonical members, so a class
  // is headed by its least member however its equalities were oriented and in
  // whatever order they arrived.
  if (precedes(b, a))
    std::swap(a, b);
  parent[b] = a;
}

llvm::DenseMap<Type, Type> TypeEquivalence::substitutionToCanonicalMembers() {
  llvm::DenseMap<Type, Type> subst;
  for (unsigned id = 0, n = terms.size(); id != n; ++id) {
    unsigned canonical = findCanonical(id);
    if (canonical != id)
      subst[terms[id]] = terms[canonical];
  }
  return subst;
}

TypeEquivalence::OrderKey &TypeEquivalence::orderKeyOf(unsigned id) {
  if (orderKeys[id])
    return *orderKeys[id];
  OrderKey key{/*projections=*/0, /*types=*/0,
               isPolymorphicType(terms[id]), std::string()};
  terms[id].walk([&](Type sub) {
    ++key.types;
    if (isa<ProjectionType>(sub))
      ++key.projections;
  });
  orderKeys[id] = std::move(key);
  return *orderKeys[id];
}

bool TypeEquivalence::precedes(unsigned a, unsigned b) {
  OrderKey &first = orderKeyOf(a);
  OrderKey &second = orderKeyOf(b);
  if (first.projections != second.projections)
    return first.projections < second.projections;
  if (first.types != second.types)
    return first.types < second.types;
  if (first.mentionsVariable != second.mentionsVariable)
    return !first.mentionsVariable;
  // Two types of the same shape are told apart by their spellings, which is the
  // one key that must print and so is printed only here.
  auto spell = [&](OrderKey &key, Type ty) -> const std::string & {
    if (key.spelling.empty()) {
      llvm::raw_string_ostream stream(key.spelling);
      stream << ty;
    }
    return key.spelling;
  };
  return spell(first, terms[a]) < spell(second, terms[b]);
}

//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

Type PolyType::specializeWith(const SpecializationMap &subst) const {
  auto self = cast<GenericTypeInterface>(*this);
  if (auto replacement = subst.lookup(self))
    return *replacement;
  return *this;
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int label = 0;

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  llvm::SMLoc labelLoc = parser.getCurrentLocation();
  if (parser.parseInteger(label)) {
    parser.emitError(parser.getNameLoc(), "expected integer");
    return Type();
  }

  // A label names a position in the declaration that binds it, so it is
  // non-negative. A negative one names no position.
  if (label < 0) {
    parser.emitError(labelLoc, "a !trait.poly label is non-negative; found ")
        << label;
    return Type();
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>'");
    return Type();
  }

  return PolyType::get(ctx, label);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getLabel() << ">";
}


//===----------------------------------------------------------------------===//
// ClaimType
//===----------------------------------------------------------------------===//

// Recover the module that anchors symbol lookups: the operation verification
// reached, or that operation itself when it is the anchoring symbol table.
ModuleOp getAnchorModule(Operation *anchor) {
  if (!anchor)
    return {};
  if (auto module = dyn_cast<ModuleOp>(anchor))
    return module;
  return anchor->getParentOfType<ModuleOp>();
}

// A claim's predicate is one of exactly two arms; the equality arm never
// carries a proof.
LogicalResult ClaimType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                                Attribute predicate, FlatSymbolRefAttr proof) {
  if (isa<TraitApplicationAttr>(predicate))
    return success();
  if (isa<TypeEqualityAttr>(predicate)) {
    if (proof)
      return emitError() << "an equality claim may not carry a proof";
    return success();
  }
  return emitError() << "claim predicate must be a trait application or a type "
                        "equality, found " << predicate;
}

// Entry point for the upstream SymbolUserTypeInterface: symbol-table
// verification invokes this for every claim reachable from an operation, and an
// owning op may call it directly. The module is recovered from the anchoring
// operation and diagnostics are anchored there.
LogicalResult ClaimType::verifySymbolUses(Operation *op,
                                          SymbolTableCollection &symbolTable) const {
  ModuleOp module = getAnchorModule(op);
  if (!module)
    return op->emitError() << "cannot verify " << *this
                           << ": anchor operation is not nested in a module";
  auto err = [&] { return op->emitError(); };

  // Equality arm: no trait symbol and no proof of its own, and the endpoints
  // are ordinary sub-elements the framework's own walk descends into, so it
  // reaches every symbol-using type nested in one and this entry point has
  // nothing left to check. The guard stands because the framework still calls
  // this on the outer equality claim, where reading the application would
  // assert.
  if (getEqualityAttr())
    return success();

  // Application arm: verify the trait application.
  if (failed(getTraitApplication().verifySymbolUses(op, symbolTable)))
    return failure();

  // if there's a proof, verify that it points to a valid symbol
  if (auto proof = getProof())
    if (failed(ProofOp::getProofOpOrUnconditionalImplOp(module, proof, err)))
      return failure();

  return success();
}

FailureOr<Attribute> parseApplicationOrEqualityPredicate(AsmParser &p) {
  MLIRContext *ctx = p.getContext();

  // The application arm opens with a trait symbol (`@Trait[...]`); anything else
  // is the equality arm (`!A = !B`), disambiguated by the leading `@`.
  FlatSymbolRefAttr traitName;
  OptionalParseResult symRes = p.parseOptionalAttribute(traitName);
  if (symRes.has_value()) {
    if (failed(*symRes))
      return failure();
    FailureOr<TraitApplicationAttr> app =
        parseTraitApplicationBody(p, traitName);
    if (failed(app))
      return failure();
    return Attribute(*app);
  }

  Type lhs, rhs;
  if (p.parseType(lhs) || p.parseEqual() || p.parseType(rhs))
    return failure();
  auto errFn = [&] { return p.emitError(p.getCurrentLocation()); };
  auto eq = TypeEqualityAttr::getChecked(errFn, ctx, lhs, rhs);
  if (!eq)
    return failure();
  return Attribute(eq);
}

Type ClaimType::parse(AsmParser& p) {
  MLIRContext *ctx = p.getContext();
  auto errFn = [&]() { return p.emitError(p.getNameLoc()); };

  if (p.parseLess())
    return {};

  FailureOr<Attribute> pred = parseApplicationOrEqualityPredicate(p);
  if (failed(pred))
    return {};

  if (auto app = dyn_cast<TraitApplicationAttr>(*pred)) {
    // An application claim may carry a `by @proof`.
    FlatSymbolRefAttr proof;
    if (succeeded(p.parseOptionalKeyword("by"))) {
      if (p.parseAttribute(proof))
        return {};
    }
    if (p.parseGreater())
      return {};
    ClaimType claim = ClaimType::getChecked(errFn, ctx, app, proof);
    return claim ? Type(claim) : Type();
  }

  // The equality arm never carries a proof; refuse `by @...` here so the
  // no-proof invariant holds at parse as well as at construction.
  auto eq = cast<TypeEqualityAttr>(*pred);
  if (succeeded(p.parseOptionalKeyword("by"))) {
    p.emitError(p.getNameLoc(),
                "an equality claim may not carry a proof");
    return {};
  }
  if (p.parseGreater())
    return {};
  ClaimType claim = ClaimType::getChecked(errFn, ctx, eq, /*proof=*/nullptr);
  return claim ? Type(claim) : Type();
}

void ClaimType::print(AsmPrinter& p) const {
  p << "<";
  if (auto eq = getEqualityAttr()) {
    eq.print(p);
  } else {
    getTraitApplication().print(p);
    if (isProven())
      p << " by " << getProof();
  }
  p << ">";
}

bool ClaimType::isPolymorphic() const {
  // An equality claim is polymorphic if either endpoint is; an application
  // claim if any of its type arguments is.
  if (auto eq = getEqualityAttr())
    return mlir::trait::isPolymorphicType(eq.getLhs()) ||
           mlir::trait::isPolymorphicType(eq.getRhs());
  return llvm::any_of(getTraitApplication().getTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  });
}

/// Verifies that two recorded proofs for the same obligation are coherent.
///
/// Proof recording keys on the demanded obligation, normalized to its ground
/// form before recording, so every path that reaches one obligation keys and
/// records it identically, and the candidate arrives already normalized at its
/// recording site. A second observation is coherent exactly when its candidate
/// equals the recorded proof literally. Any residual disagreement -- a
/// different proof symbol, or a spelling that does not match after
/// normalization -- is an incoherent proof mapping.
static LogicalResult verifyEquivalentRecordedProof(
    ClaimType unproven,
    ClaimType recorded,
    ClaimType candidate,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (recorded == candidate)
    return success();

  if (err) err() << "inconsistent proof mapping: " << unproven
                 << " is already bound to " << recorded
                 << ", but attempted to bind " << candidate;
  return failure();
}

namespace {

/// What one node of a derivation produced.
///
/// A node's closure is its own binding followed by its children's, which is
/// what replaying it into another evidence map has to write. A node is complete
/// when this derivation computed all of that: a child that exited early on a
/// binding this derivation did not itself write contributes a closure nobody
/// here knows, and neither it nor anything above it can be held.
struct DerivedNode {
  ProofDerivationMemo::Closure closure;
  bool complete = true;

  /// Adds one binding, which a node already carrying it has already written.
  ///
  /// A closure is the SET of bindings replaying it writes, kept in derivation
  /// order for a reader. One obligation can be reached through two of a proof's
  /// subtrees, and whether the second reaching writes it again or exits early on
  /// the first is decided by the order the caller's own map was filled in -- so
  /// keeping a binding once is what makes two derivations of one pair produce
  /// one closure.
  void add(ClaimType unproven, ClaimType proven) {
    if (written.insert(std::make_pair(unproven, proven)).second)
      closure.emplace_back(unproven, proven);
  }

  void addAll(const ProofDerivationMemo::Closure &other) {
    for (auto [unproven, proven] : other)
      add(unproven, proven);
  }

  /// Takes `other` as this node's whole closure, which a node that wrote
  /// nothing of its own does when another derivation already holds it.
  void take(const ProofDerivationMemo::Closure &other) {
    closure.clear();
    written.clear();
    addAll(other);
  }

private:
  llvm::DenseSet<std::pair<ClaimType, ClaimType>> written;
};

/// The nodes one top-level derivation has completed, held until it succeeds.
///
/// Nothing is put in the memo while the derivation that produced it is still
/// running. A node reached through an ancestor's optimistic binding was derived
/// under an assumption that ancestor can still take back, and the map the
/// derivation writes into is rolled back with it; publishing on the outermost
/// success is what keeps the memo from outliving an assumption that failed.
///
/// A node is also what a later node of the same derivation exits early on, so
/// this is indexed by the normalized obligation the early exit looks up as well
/// as by the pair the memo is keyed on.
class DerivationStaging {
public:
  void hold(ClaimType keyUnproven, ClaimType keyProven,
            ClaimType normalizedUnproven, ClaimType normalizedProven,
            const ProofDerivationMemo::Closure &closure) {
    byNormalizedObligation[normalizedUnproven] = held.size();
    held.push_back(
        Held{keyUnproven, keyProven, normalizedUnproven, normalizedProven,
             closure});
  }

  /// What deriving the obligation now bound to `normalizedUnproven` produced,
  /// when this derivation is what bound it.
  const ProofDerivationMemo::Closure *
  lookupDerived(ClaimType normalizedUnproven) const {
    auto it = byNormalizedObligation.find(normalizedUnproven);
    if (it == byNormalizedObligation.end())
      return nullptr;
    return &held[it->second].closure;
  }

  /// Publishes every node into the memo of spelling pairs, and every node's
  /// closure into the record of what deriving its pair produces. One derivation
  /// reads one module, which both are keyed by along with the pair.
  ///
  /// The record decides for itself what it can keep: an unsettled derivation and
  /// a pair two derivations disagree over are both refused there.
  void publishInto(ProofDerivationMemo &memo, ModuleOp module) const {
    ProofClosureRecord &closures = memo.getClosures();
    for (const Held &node : held) {
      memo.record(module, node.keyUnproven, node.keyProven, node.closure);
      (void)closures.record(module, node.normalizedUnproven,
                            node.normalizedProven, node.closure);
    }
  }

private:
  struct Held {
    ClaimType keyUnproven;
    ClaimType keyProven;
    ClaimType normalizedUnproven;
    ClaimType normalizedProven;
    ProofDerivationMemo::Closure closure;
  };

  SmallVector<Held, 8> held;
  llvm::DenseMap<ClaimType, size_t> byNormalizedObligation;
};

} // namespace

static LogicalResult deriveProof(ClaimType unproven, ClaimType proven,
                                 ModuleOp module, EvidenceBindings &bindings,
                                 DemandOrigin origin,
                                 ProofDerivationMemo *memo,
                                 DerivationStaging &staging,
                                 DerivedNode &derived,
                                 llvm::function_ref<InFlightDiagnostic()> err);

/// Writes a closure a derivation already produced into `bindings`.
///
/// Every entry is looked up before it is written, because a differing
/// re-binding is a program the compiler must diagnose rather than an
/// impossibility it may assume: the same obligation can arrive proven by two
/// symbols, and that is the incoherent proof mapping the derivation this
/// replaces reports at its own early exit.
static LogicalResult replayClosure(const ProofDerivationMemo::Closure &closure,
                                   EvidenceBindings &bindings,
                                   llvm::function_ref<InFlightDiagnostic()> err) {
  for (auto [unproven, proven] : closure) {
    if (auto existing = bindings.lookup(unproven)) {
      if (failed(verifyEquivalentRecordedProof(unproven, *existing, proven, err)))
        return failure();
      continue;
    }
    bindings.bind(unproven, proven);
  }
  return success();
}

/// Derives one node of a proof, extending `bindings` with everything the node's
/// own claim and its obligations bind.
static LogicalResult deriveProof(ClaimType unproven, ClaimType proven,
                                 ModuleOp module, EvidenceBindings &bindings,
                                 DemandOrigin origin,
                                 ProofDerivationMemo *memo,
                                 DerivationStaging &staging,
                                 DerivedNode &derived,
                                 llvm::function_ref<InFlightDiagnostic()> err) {
  // the proven side must carry a proof
  if (!proven.isProven()) {
    if (err) err() << "expected proven claim, but found " << proven;
    return failure();
  }

  // the unproven side must be an unproven obligation: it is the recording key,
  // and a proven claim here would trip the bindings.bind precondition
  // downstream. Reject it with a diagnostic instead of reaching that assert.
  if (unproven.isProven()) {
    if (err) err() << "expected unproven obligation, but found proven claim "
                   << unproven;
    return failure();
  }

  // The memo answers for the pair as it arrived, before either side is
  // normalized: that is the pair a caller holds, and answering here is what
  // skips the two normalizations below as well as the derivation under them.
  ClaimType askedUnproven = unproven;
  ClaimType askedProven = proven;
  if (memo) {
    if (const auto *closure = memo->lookup(module, askedUnproven, askedProven)) {
      derived.take(*closure);
      return replayClosure(*closure, bindings, err);
    }
  }

  // Normalize both the demanded obligation (the recording key) and the proven
  // value before recording. Requirement obligations arrive at their stamped
  // declaration projections; resolving those ground projections means every path
  // that reaches the same obligation keys it identically and records the same
  // proven spelling, so a second observation matches the first literally
  // instead of reconciling two equivalent spellings.
  {
    // A cyclic associated-type binding leaves these ground projections without
    // a normal form. The fallible resolver refuses it here so proof
    // verification fails cleanly on hostile IR rather than the resolution
    // running the process out of its budget deeper down.
    // XXX TODO a projection a declaration spells must be over its own self
    // application, a where-clause application, a trait requirement or a declared
    // witness (Rust's projection well-formedness rule), so every projection has
    // evidence at a known index and this module read deletes with LookupScope and
    // the verifier DemandOrigins.
    FailureOr<Type> normalizedProven =
        resolveProjectionsByLookup(proven, module, origin,
                                   LookupScope::Ground, err);
    if (failed(normalizedProven))
      return failure();
    proven = cast<ClaimType>(*normalizedProven);

    FailureOr<Type> normalizedUnproven =
        resolveProjectionsByLookup(unproven, module, origin,
                                   LookupScope::Ground, err);
    if (failed(normalizedUnproven))
      return failure();
    unproven = cast<ClaimType>(*normalizedUnproven);
  }

  // An obligation is discharged only by evidence for that same application.
  // The evidence's claim is a declaration over the variables it spells -- a
  // blanket impl and a proof written over type variables each stand for every
  // instance of theirs -- so the judgment is whether that declaration, read at
  // the arguments this obligation supplies, rebuilds the obligation. Everything
  // below reads the evidence's own header and its own subproofs, so without
  // this a citation of an impl or a proof of some other application would be
  // checked against itself and pass, and an obligation spelling a variable
  // would take evidence for one instance of it as evidence for all of them.
  {
    // XXX TODO a projection a declaration spells must be over its own self
    // application, a where-clause application, a trait requirement or a declared
    // witness (Rust's projection well-formedness rule), so every projection has
    // evidence at a known index and this module read deletes with LookupScope and
    // the verifier DemandOrigins.
    GroundProjectionLookup byGroundLookup(module, origin);
    if (failed(matchDeclaration(getTypeParametersIn(Type(proven)),
                                Type(proven.asUnproven()), Type(unproven),
                                byGroundLookup, /*err=*/nullptr))) {
      // A side still spelling a projection is one nothing here can decide: the
      // impls standing now resolve it for nobody, and impl selection resolves
      // it through the candidate it settles on, which a reader holding no
      // record cannot. The obligation is declined rather than discharged --
      // nothing is bound, and this node describes no closure -- so the claim
      // stands unproven for selection to derive and for the leftover walk to
      // refuse.
      if (containsType<ProjectionType>(Type(unproven)) ||
          containsType<ProjectionType>(Type(proven))) {
        derived.complete = false;
        return success();
      }
      if (err) err() << "proof " << proven.getProof() << " proves "
                     << proven.asUnproven()
                     << ", which does not discharge the obligation " << unproven;
      return failure();
    }
  }

  // What deriving one settled pair produces is a fact about the proof standing
  // over it and not about the caller that reached it, so a pair the record
  // already holds is replayed here instead of derived a second time. This is
  // the node every reader shares: a call site asking about a proven claim, and
  // an obligation underneath some other derivation, both arrive at this pair in
  // the grade the record is keyed in, and both get the closure the derivation
  // that ran first wrote. Replaying it writes exactly the bindings deriving
  // would write, including the pair's own, so nothing below needs to run.
  if (memo) {
    if (const auto *closure =
            memo->getClosures().lookup(module, unproven, proven)) {
      derived.take(*closure);
      return replayClosure(*closure, bindings, err);
    }
  }

  // early exit if we've already recorded this obligation. The same proof may
  // be observed through multiple equivalent claim spellings, so validate proof
  // coherence instead of requiring syntactic claim equality.
  if (auto existing = bindings.lookup(unproven)) {
    if (failed(verifyEquivalentRecordedProof(unproven, *existing, proven, err)))
      return failure();
    // What this node would have written is already written. When this
    // derivation is what wrote it, that closure is in hand and stands for this
    // node's; when something before this derivation wrote it, the record of
    // what deriving the application produces is where the closure is, because
    // it is kept per application rather than per derivation. Only where neither
    // has it does the node go undescribed, and nothing containing it can be
    // held either.
    if (const auto *closure = staging.lookupDerived(unproven)) {
      derived.take(*closure);
    } else if (const auto *recorded =
                   memo ? memo->getClosures().lookup(module, unproven, proven)
                        : nullptr) {
      derived.take(*recorded);
    } else {
      derived.complete = false;
    }
    return success();
  }

  // look up the trait and its requirements using the unproven claim
  auto trait = unproven.getTraitApplication().getTrait(module, err);
  if (failed(trait)) return failure();

  // inspect the proof symbol on the proven side
  auto symOp = ProofOp::getProofOpOrUnconditionalImplOp(module, proven.getProof(), err);
  if (failed(symOp)) return failure();

  // if it's an impl op, check that the trait has no requirements
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (trait->hasRequirements()) {
      if (err) err() << "impl provides no subproof for trait requirements";
      return failure();
    }

    // Naming an unconditional impl is not the same as proving this claim: the
    // proof could cite an impl of a different trait, or of this trait at
    // arguments the claim does not meet, and nothing above has compared the two.
    // Match the impl's own header against the proven claim -- the same citation
    // check that verifying a witness runs -- so a proof whose impl cannot be
    // carried to its claim is refused here rather than trusted to a leaf.
    // XXX TODO a projection a declaration spells must be over its own self
    // application, a where-clause application, a trait requirement or a declared
    // witness (Rust's projection well-formedness rule), so every projection has
    // evidence at a known index and this module read deletes with LookupScope and
    // the verifier DemandOrigins.
    GroundProjectionLookup byGroundLookup(module, origin);
    if (failed(impl.buildSubstitutionForSelfClaim(proven, byGroundLookup, err)))
      return failure();

    // success: bind the whole claim so that later normalization keeps the proof
    bindings.bind(unproven, proven);
    // A leaf: the binding it wrote is the whole of what deriving it produces.
    derived.add(unproven, proven);
    staging.hold(askedUnproven, askedProven, unproven, proven, derived.closure);
    return success();
  }

  // otherwise the symbol must be a ProofOp
  auto proof = dyn_cast<ProofOp>(*symOp);

  // The proof's own claim is the declaration and the claim it is cited for is
  // the use: a proof op may be written over type variables and stand for every
  // instance of them, so its parameters take the arguments the cited claim
  // supplies and the claim it rebuilds must be that citation.
  {
    Type proofClaim = Type(proof.getProvenClaim());
    // XXX TODO a projection a declaration spells must be over its own self
    // application, a where-clause application, a trait requirement or a declared
    // witness (Rust's projection well-formedness rule), so every projection has
    // evidence at a known index and this module read deletes with LookupScope and
    // the verifier DemandOrigins.
    GroundProjectionLookup byGroundLookup(module, origin);
    if (failed(matchDeclaration(getTypeParametersIn(proofClaim), proofClaim,
                                Type(proven), byGroundLookup, err)))
      return failure();
  }

  // The impl's obligations are read at the claim this proof is cited for, which
  // the match above carried the proof's declaration to. A proof written over
  // type variables states its obligations over those same variables, and the
  // instance a citation supplies is what they stand at here: read at the
  // declaration instead, an obligation would be discharged by evidence for
  // whichever instance the proof's own subproof happened to name.
  //
  // That claim is the one whose projections are resolved, which the obligation
  // this node was asked about may still spell. Example: unproven =
  // @D[A[i32]::Out, A[f32]::Out], cited = @D[i64, i64]. An impl like
  // @D[poly, poly] rebuilds @D[i64, i64] but not @D[A[i32]::Out, A[f32]::Out]
  // (the two projections are structurally different even though both resolve to
  // i64).
  auto obligations = proof.getImpl().specializeObligationsAsClaimsFor(
      proven.asUnproven(), origin, err);
  if (failed(obligations)) return failure();

  // get the subproof claims (also checks arity against obligations)
  auto subproofs = proof.verifyAndGetSubproofClaims(origin, err);
  if (failed(subproofs)) return failure();

  // Bind optimistically before recursing so that coinductive self-references
  // (where an obligation resolves back to the same claim) hit the early exit
  // at the top of this function instead of diverging.
  bindings.bind(unproven, proven);
  derived.add(unproven, proven);

  // recurse over obligations
  for (auto [ob, sub] : llvm::zip(*obligations, *subproofs)) {
    DerivedNode child;
    if (failed(deriveProof(ob, sub, module, bindings, origin, memo, staging,
                           child, err))) {
      bindings.erase(unproven);
      return failure();
    }
    derived.complete &= child.complete;
    if (derived.complete)
      derived.addAll(child.closure);
  }

  if (derived.complete) {
    staging.hold(askedUnproven, askedProven, unproven, proven, derived.closure);
  } else {
    derived.take({});
  }
  return success();
}

bool mentionsMonomorphicProjection(Type ty) {
  bool found = false;
  DenseSet<Type> seen;

  auto visit = [&](Type node, auto &visitRef) -> void {
    if (found || !seen.insert(node).second)
      return;
    if (auto projection = dyn_cast<ProjectionType>(node)) {
      if (isMonomorphicType(projection)) {
        found = true;
        return;
      }
      for (Type arg : projection.getTraitApplication().getTypeArgs())
        visitRef(arg, visitRef);
      for (Type arg : projection.getAssocTypeArgs())
        visitRef(arg, visitRef);
    } else if (auto claim = dyn_cast<ClaimType>(node)) {
      for (Type arg : claim.getTraitApplication().getTypeArgs())
        visitRef(arg, visitRef);
    }

    node.walkImmediateSubElements(
        /*walkAttrsFn=*/[](Attribute) {},
        /*walkTypesFn=*/[&](Type subTy) { visitRef(subTy, visitRef); });
  };

  visit(ty, visit);
  return found;
}

LogicalResult verifyAndRecordProof(
    ClaimType unproven,
    ClaimType proven,
    ModuleOp module,
    EvidenceBindings &bindings,
    DemandOrigin origin,
    ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DerivationStaging staging;
  DerivedNode derived;
  if (failed(deriveProof(unproven, proven, module, bindings, origin, memo,
                         staging, derived, err)))
    return failure();

  // Everything this derivation completed goes into the memo together, now that
  // the derivation it was completed under has returned success.
  if (memo)
    staging.publishInto(*memo, module);
  return success();
}

/// Walk `root` and record substitution entries for every proven claim
/// found within it. This maps the unproven claim to the proven claim, which is
/// what lets a call substitution respell a claim a spelling names to the
/// spelling that carries its evidence.
LogicalResult bindProofsIn(
    Type root,
    ModuleOp module,
    EvidenceBindings &bindings,
    DemandOrigin origin,
    ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  LogicalResult status = success();

  root.walk([&](Type node) {
    if (status.failed()) return;

    if (auto claim = dyn_cast<ClaimType>(node)) {
      if (claim.isProven()) {
        if (failed(verifyAndRecordProof(claim.asUnproven(), claim, module,
                                        bindings, origin, memo, err)))
          status = failure();
      }
    }
  });

  return status;
}

void ClaimType::getProjections(
    ModuleOp module,
    SmallVectorImpl<ClaimType>& result) {
  // Equality claims are not projected from; they are consumed by trait.coerce.
  if (isEquality())
    return;

  // identity
  result.push_back(*this);

  // trait requirements
  auto trait = getTraitApplication().getTraitOrAbort(module, "ClaimType::getProjections: couldn't find trait");
  auto specRequirements = trait.specializeRequirementsAsClaimsFor(*this);
  if (succeeded(specRequirements))
    result.append(*specRequirements);

  // A proven source additionally projects to each of its obligations spelled
  // proven by the subproof discharging it, and to the impl's equality
  // where-clauses. An unproven source keeps only the unproven candidates above:
  // proofness parity refuses a proven application result projected from it.
  if (isProven()) {
    if (auto proof = SymbolTable::lookupNearestSymbolFrom<ProofOp>(module, getProof())) {
      ImplOp impl = proof.getImpl();
      if (impl) {
        // XXX TODO a projection a declaration spells must be over its own self
        // application, a where-clause application, a trait requirement or a
        // declared witness (Rust's projection well-formedness rule), so every
        // projection has evidence at a known index and this module read deletes
        // with LookupScope and the verifier DemandOrigins.
        GroundProjectionLookup byGroundLookup(
            module, DemandOrigin::ProofVerification);

        // Unproven impl assumptions, kept so the candidate query does not lose
        // a candidate; the proven spelling below supersedes them.
        auto specAssumptions = impl.specializeAssumptionsAsClaimsFor(
            *this, byGroundLookup, /*errFn=*/nullptr);
        if (succeeded(specAssumptions))
          result.append(*specAssumptions);

        // The impl's obligations in the order the proof's subproof names align
        // with (the trait's requirements then the impl's assumptions), each
        // spelled proven by its subproof. This runs in a verifier, so it reads
        // the proof structure at a non-recording origin.
        auto obligations = impl.specializeObligationsAsClaimsFor(
            *this, DemandOrigin::ProofVerification, /*errFn=*/nullptr);
        ArrayAttr subproofNames = proof.getSubproofNames();
        if (succeeded(obligations) &&
            subproofNames.size() == obligations->size())
          for (auto [ob, name] : llvm::zip(*obligations, subproofNames))
            if (auto ref = dyn_cast<FlatSymbolRefAttr>(name))
              result.push_back(
                  ClaimType::get(getContext(), ob.getTraitApplication(), ref));

        // The impl's equality where-clauses, specialized for this source. An
        // equality claim never carries a proof, so it is a parity-exempt
        // candidate an equality projection resolves to.
        auto eqSubst = impl.buildSubstitutionForSelfClaim(
            *this, byGroundLookup, /*errFn=*/nullptr);
        if (succeeded(eqSubst)) {
          for (Attribute pred : impl.getAssumptions())
            if (auto eq = dyn_cast<TypeEqualityAttr>(pred))
              result.push_back(ClaimType::getEquality(
                  getContext(), instantiate(eq.getLhs(), *eqSubst),
                  instantiate(eq.getRhs(), *eqSubst)));
        }
      }
    }
  }
}

bool ClaimType::projectsTo(ModuleOp module, ClaimType dst) {
  SmallVector<ClaimType> candidates;
  getProjections(module, candidates);
  for (ClaimType cand : candidates)
    if (cand == dst)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ProjectionType
//===----------------------------------------------------------------------===//

bool ProjectionType::isPolymorphic() const {
  return llvm::any_of(getTraitApplication().getTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  }) || llvm::any_of(getAssocTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  });
}

Type ProjectionType::parse(AsmParser &p) {
  MLIRContext *ctx = p.getContext();

  if (p.parseLess())
    return {};

  // parse @Trait[Types...]
  TraitApplicationAttr app = mlir::dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app)
    return {};

  if (p.parseComma())
    return {};

  // parse "AssocName"
  StringAttr assocName;
  if (p.parseAttribute(assocName))
    return {};

  // parse optional , [gat_args...]
  SmallVector<Type> assocTypeArgs;
  if (succeeded(p.parseOptionalComma())) {
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
          Type ty;
          if (p.parseType(ty)) return failure();
          assocTypeArgs.push_back(ty);
          return success();
        })))
      return {};
  }

  if (p.parseGreater())
    return {};

  return ProjectionType::get(ctx, app, assocName, assocTypeArgs);
}

void ProjectionType::print(AsmPrinter &p) const {
  p << "<";
  getTraitApplication().print(p);
  p << ", " << getAssocName();
  if (!getAssocTypeArgs().empty()) {
    p << ", [";
    llvm::interleaveComma(getAssocTypeArgs(), p, [&](Type ty) {
      p.printType(ty);
    });
    p << "]";
  }
  p << ">";
}

// Entry point for the upstream SymbolUserTypeInterface. A projection carries the
// same trait application as its claim, so verification delegates to that claim.
LogicalResult ProjectionType::verifySymbolUses(Operation *op,
                                               SymbolTableCollection &symbolTable) const {
  return asClaim().verifySymbolUses(op, symbolTable);
}

//===----------------------------------------------------------------------===//
// Declaration matching
//===----------------------------------------------------------------------===//

unsigned firstUnusedPolyLabel(Operation *op) {
  // The declaration `op` stands in: the outermost operation below the module,
  // which is what a substitution over this code is keyed by.
  Operation *scope = op;
  for (Operation *parent = op->getParentOp();
       parent && !isa<ModuleOp>(parent); parent = parent->getParentOp())
    scope = parent;

  unsigned next = 0;
  auto readType = [&](Type ty) {
    // Claims and projections hold their type arguments in an attribute, so the
    // reader that descends through those is the one that sees every label.
    for (GenericTypeInterface generic : getGenericTypesIn(ty))
      if (auto poly = dyn_cast<PolyType>(generic.getParameterAtom()))
        next = std::max(next, static_cast<unsigned>(poly.getLabel()) + 1);
  };
  auto readAttribute = [&](Attribute attr) {
    attr.walk([&](Attribute sub) {
      if (auto typeAttr = dyn_cast<TypeAttr>(sub))
        readType(typeAttr.getValue());
    });
  };
  scope->walk([&](Operation *op) {
    for (Type ty : op->getResultTypes())
      readType(ty);
    for (NamedAttribute attr : op->getAttrs())
      readAttribute(attr.getValue());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          readType(arg.getType());
  });
  return next;
}

LogicalResult TypeArguments::assign(
    GenericTypeInterface parameter, Type value,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto index = indexOf(parameter);
  if (!index) {
    if (err)
      err() << "type parameter " << Type(parameter)
            << " is not bound by this declaration";
    return failure();
  }
  std::optional<Type> &slot = slots[*index];
  if (!slot) {
    slot = value;
    return success();
  }
  if (*slot == value)
    return success();
  if (err)
    err() << "conflicting type arguments for " << Type(parameter) << ": "
          << *slot << " versus " << value;
  return failure();
}

namespace {

/// One pass of the reading. `throughProjections` says whether a formal
/// projection may be read: a parameter standing only inside one is determined
/// by the position the projection itself stands in, so the first pass leaves
/// projections alone and the second reads what the first did not fill.
void extractInto(Type formal, Type actual, TypeArguments &args,
                 bool throughProjections) {
  // A parameter of this declaration takes whatever stands opposite it. This
  // comes first so that a parameter matched against itself is still recorded.
  // A second, differing reading of one parameter keeps the first: the reading
  // runs before either side is normalized, so two spellings that differ here
  // may yet be one type.
  if (GenericTypeInterface parameter = getParameterOccurrence(formal)) {
    // A parameter this declaration does not bind is rigid, and so is whatever
    // spelling carries it: nothing under one is read, and whether the two sides
    // agree there is the comparison's question.
    if (args.binds(parameter))
      (void)args.assign(parameter, actual, /*err=*/nullptr);
    return;
  }

  // A projection is not injective, so nothing is read out of the position it
  // stands in. Its arguments are read only where the actual side spells the
  // same projection, and only after every position outside a projection has had
  // its say -- what a projection's arguments confirm, a position outside one
  // determines.
  if (isa<ProjectionType>(formal) && !throughProjections)
    return;

  // Every other node is read by its shape: the same constructor, the same
  // attributes and the same child count, then the children by position. A
  // claim's predicate and a projection's arguments are enumerated here as
  // children, and an application claim's key ignores its proof, so the reading
  // is blind to the evidence either side carries.
  //
  // Two shapes that differ are not a refusal either: a position where they
  // diverge is one this has nothing to learn from -- the actual may yet
  // normalize into the formal's shape. Whatever the reading leaves unfilled
  // stands in the rebuilt declaration, and the comparison is what refuses.
  TermShape formalShape = decomposeTerm(formal);
  TermShape actualShape = decomposeTerm(actual);
  if (formalShape.key != actualShape.key ||
      formalShape.children.size() != actualShape.children.size())
    return;
  for (auto [formalChild, actualChild] :
       llvm::zip(formalShape.children, actualShape.children))
    extractInto(formalChild, actualChild, args, throughProjections);
}

} // namespace

void extractTypeArguments(Type formal, Type actual, TypeArguments &args) {
  extractInto(formal, actual, args, /*throughProjections=*/false);
  extractInto(formal, actual, args, /*throughProjections=*/true);
}

LogicalResult verifyEqualAfterInstantiation(
    Type formal, const SpecializationMap &args, Type actual,
    Normalizer normalize, llvm::function_ref<InFlightDiagnostic()> err) {
  Type rebuilt = stripClaimProofs(instantiate(formal, args));
  Type wanted = stripClaimProofs(actual);
  if (normalize) {
    FailureOr<Type> normalizedRebuilt = normalize(rebuilt);
    if (failed(normalizedRebuilt))
      return failure();
    FailureOr<Type> normalizedWanted = normalize(wanted);
    if (failed(normalizedWanted))
      return failure();
    rebuilt = *normalizedRebuilt;
    wanted = *normalizedWanted;
  }
  if (rebuilt == wanted)
    return success();
  if (err)
    err() << "type mismatch: expected " << rebuilt << " but found " << wanted;
  return failure();
}

FailureOr<SpecializationMap> matchDeclaration(
    ArrayRef<GenericTypeInterface> parameters, Type formal, Type actual,
    Normalizer normalize, llvm::function_ref<InFlightDiagnostic()> err) {
  TypeArguments args(parameters);
  extractTypeArguments(formal, actual, args);
  SpecializationMap specialization = args.toSpecialization();
  if (failed(verifyEqualAfterInstantiation(formal, specialization, actual,
                                           normalize, err)))
    return failure();
  return specialization;
}

} // end mlir::trait
