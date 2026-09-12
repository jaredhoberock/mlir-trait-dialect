// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "DemandLedger.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <atomic>
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
    concreteTypes.push_back(applySubstitutionToFixedPoint(subst, ty));
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
/// [invariant] A speculative cross-check raises no diagnostic. Under a
/// `DemandCrossCheckScope` the caller resolves only to compare the spelling and
/// then discards it, so a non-terminating projection there is a silent decline,
/// not a user-facing error; `isCrossChecking()` enforces the suppression.
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
  if (isCrossChecking())
    return;
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
  // declines on it in the safe direction; the reporter surfaces the diagnostic
  // for a live demand and stays silent under a cross-check.
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
  // reached from untrusted IR fails cleanly rather than admitting the cycle. A
  // speculative cross-check discards the spelling, so it declines silently.
  if (!converged) {
    if (emitError && !isCrossChecking())
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

int nextPolyTypeId() {
  static std::atomic<int> counter{-1};
  return counter.fetch_sub(1, std::memory_order_relaxed);
}

PolyType PolyType::getUnique(MLIRContext* ctx) {
  return PolyType::get(ctx, nextPolyTypeId());
}

Type PolyType::instantiate(InstantiationMap &inst, uint64_t &idCounter) {
  auto self = cast<GenericTypeInterface>(*this);

  // check memo first - if we've already instantiated this PolyType, return it
  if (auto existing = inst.lookup(self))
    return *existing;

  // create and remember a fresh inference var for this poly
  auto fresh = InferenceType::get(getContext(), idCounter++);
  inst.bind(self, cast<UnificationTypeInterface>(fresh));
  return fresh;
}

Type PolyType::specializeWith(const SpecializationMap &subst) const {
  auto self = cast<GenericTypeInterface>(*this);
  if (auto replacement = subst.lookup(self))
    return *replacement;
  return *this;
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int uniqueId = 0;

  // parse this:
  // <unique> or
  // <int>

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  if (succeeded(parser.parseOptionalKeyword("unique"))) {
    uniqueId = nextPolyTypeId();
  } else {
    if (parser.parseInteger(uniqueId)) {
      parser.emitError(parser.getNameLoc(), "expected integer or 'unique'");
      return Type();
    }
    
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>'");
    return Type();
  }

  return PolyType::get(ctx, uniqueId);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getUniqueId() << ">";
}


//===----------------------------------------------------------------------===//
// InferenceType
//===----------------------------------------------------------------------===//

LogicalResult InferenceType::unify(
  Type other,
  ModuleOp /*module*/,
  UnificationMap &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;
  auto selfKey = cast<UnificationTypeInterface>(self);

  // normalize
  other = applySubstitutionOnce(subst.toTypeMap(), other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto existing = subst.lookup(selfKey)) {
    if (*existing != other) {
      if (err) return err() << "inference variable " << self
                            << " already bound to " << *existing
                            << ", cannot bind to " << other;
      return failure();
    }
    return success();
  }

  // occurs check: forbid T := f(..., T, ...) to avoid cycles
  auto occursIn = [](Type needle, Type haystack) {
    bool hit = false;
    haystack.walk([&](Type t) {
      if (!hit && t == needle) hit = true;
    });
    return hit;
  };

  if (occursIn(self, other)) {
    if (err) err() << "recursive substitution: " << self
                   << " occurs in " << other;
    return failure();
  }

  // bind the variable
  subst.bind(selfKey, other);
  return success();
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
  /// closure into the record of what deriving its pair produces.
  ///
  /// The record decides for itself what it can keep: an unsettled derivation and
  /// a pair two derivations disagree over are both refused there.
  void publishInto(ProofDerivationMemo &memo) const {
    ProofClosureRecord &closures = memo.getClosures();
    for (const Held &node : held) {
      memo.record(node.keyUnproven, node.keyProven, node.closure);
      (void)closures.record(node.normalizedUnproven, node.normalizedProven,
                            node.closure);
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
    if (const auto *closure = memo->lookup(askedUnproven, askedProven)) {
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

  // What deriving one settled pair produces is a fact about the proof standing
  // over it and not about the caller that reached it, so a pair the record
  // already holds is replayed here instead of derived a second time. This is
  // the node every reader shares: a call site asking about a proven claim, and
  // an obligation underneath some other derivation, both arrive at this pair in
  // the grade the record is keyed in, and both get the closure the derivation
  // that ran first wrote. Replaying it writes exactly the bindings deriving
  // would write, including the pair's own, so nothing below needs to run.
  if (memo) {
    if (const auto *closure = memo->getClosures().lookup(unproven, proven)) {
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
                   memo ? memo->getClosures().lookup(unproven, proven)
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
    GroundProjectionLookup byGroundLookup(module, origin);
    if (failed(matchDeclaration(getTypeParametersIn(proofClaim), proofClaim,
                                Type(proven), byGroundLookup, err)))
      return failure();
  }

  // Use the proof's concrete claim (projections resolved) rather than the
  // unproven claim (which may still contain projections). Example:
  // unproven = @D[A[i32]::Out, A[f32]::Out], concrete = @D[i64, i64].
  // An impl like @D[poly, poly] can unify with @D[i64, i64] but not with
  // @D[A[i32]::Out, A[f32]::Out] (the two projections are structurally
  // different even though both resolve to i64).
  auto obligations = proof.getImpl().specializeObligationsAsClaimsFor(
      proof.getProvenClaim().asUnproven(), origin, err);
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
    staging.publishInto(*memo);
  return success();
}

/// Walk `root` and record substitution entries for every proven claim
/// found within it.  This maps the unproven claim to the proven claim.
/// These entries are used during unification so that
/// `applySubstitutionToFixedPoint` can normalize claims before
/// per-type unification dispatch.
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
          auto substMap = eqSubst->toTypeMap();
          for (Attribute pred : impl.getAssumptions())
            if (auto eq = dyn_cast<TypeEqualityAttr>(pred))
              result.push_back(ClaimType::getEquality(
                  getContext(),
                  applySubstitutionToFixedPoint(substMap, eq.getLhs()),
                  applySubstitutionToFixedPoint(substMap, eq.getRhs())));
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

static LogicalResult unifyTypeRange(ArrayRef<Type> formalTypes,
                                    ArrayRef<Type> actualTypes,
                                    ModuleOp module,
                                    UnificationMap &subst,
                                    llvm::function_ref<InFlightDiagnostic()> err);

LogicalResult ClaimType::unify(
    Type other,
    ModuleOp module,
    UnificationMap& subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize formal first
  Type formalNormTy = applySubstitutionOnce(subst.toTypeMap(), *this);
  ClaimType formal = mlir::dyn_cast<ClaimType>(formalNormTy);

  // if formal is no longer a ClaimType, delegate to generic path
  if (!formal)
    return trait::unify(formalNormTy, other, module, subst, err);

  // normalize actual second
  Type normActualTy = applySubstitutionOnce(subst.toTypeMap(), other);
  ClaimType actual = mlir::dyn_cast<ClaimType>(normActualTy);

  // if actual isn't a claim, it's an immediate mismatch
  if (!actual) {
    if (err) {
      err() << "expected !trait.claim, but found " << normActualTy;
    }
    return failure();
  }

  // do claim-specific checks below

  // Arm dispatch. A claim of one arm never unifies with the other.
  if (auto formalEq = formal.getEqualityAttr()) {
    auto actualEq = actual.getEqualityAttr();
    if (!actualEq) {
      if (err) err() << "expected an equality claim, but found " << actual;
      return failure();
    }
    // Endpoint-wise unification through the ordinary substitution machinery:
    // variable binding, no leniency, no projection tolerance, no proof
    // sensitivity (the equality arm carries none). Orientation is fixed, so
    // lhs matches lhs and rhs matches rhs.
    if (failed(trait::unify(formalEq.getLhs(), actualEq.getLhs(), module, subst, err)))
      return failure();
    return trait::unify(formalEq.getRhs(), actualEq.getRhs(), module, subst, err);
  }
  if (actual.getEqualityAttr()) {
    if (err) err() << "expected a trait-application claim, but found " << actual;
    return failure();
  }

  auto formalApp = formal.getTraitApplication();
  auto actualApp = actual.getTraitApplication();

  // same trait?
  if (formalApp.getTraitName() != actualApp.getTraitName()) {
    if (err) err() << "trait mismatch: expected " << formalApp.getTraitName()
                   << ", but found " << actualApp.getTraitName();
    return failure();
  }

  // check proofs
  auto formalProof = formal.getProof();
  auto actualProof = actual.getProof();
  if (formalProof && actualProof && formalProof != actualProof) {
    if (err) err() << "proof mismatch: expected " << formalProof
                   << ", but found " << actualProof;
    return failure();
  }
  if (formalProof && !actualProof) {
    if (err) err() << "cannot unify proven claim with unproven claim";
    return failure();
  }

  return unifyTypeRange(formalApp.getTypeArgs(), actualApp.getTypeArgs(), module,
                        subst, err);
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
// unify
//===----------------------------------------------------------------------===//

static LogicalResult unifyTypeRange(ArrayRef<Type> formalTypes,
                                    ArrayRef<Type> actualTypes,
                                    ModuleOp module,
                                    UnificationMap &subst,
                                    llvm::function_ref<InFlightDiagnostic()> err) {
  if (formalTypes.size() != actualTypes.size()) {
    if (err)
      err() << "type arity mismatch: expected " << formalTypes.size()
            << " type arguments, but found " << actualTypes.size();
    return failure();
  }

  for (auto [formal, actual] : llvm::zip(formalTypes, actualTypes)) {
    if (failed(trait::unify(formal, actual, module, subst, err)))
      return failure();
  }

  return success();
}

/// Collect exactly the immediate child Types and Attributes of `ty`. If `ty` has no sub‐elements,
/// returns empty vectors.
static std::pair<SmallVector<Type, 4>, SmallVector<Attribute, 4>> getImmediateSubElements(Type ty) {
  SmallVector<Type, 4> childTypes;
  SmallVector<Attribute, 4> childAttrs;
  ty.walkImmediateSubElements(
      /*walkAttrsFn=*/[&](Attribute subAttr) {
        childAttrs.push_back(subAttr);
      },
      /*walkTypesFn=*/[&](Type subTy) {
        childTypes.push_back(subTy);
      });
  return std::pair(childTypes, childAttrs);
}

/// Whether `ty` carries a projection all of whose arguments are concrete, i.e.
/// one a unique module-visible impl could resolve. Only such a projection lets
/// the resolve-then-rebuild step in `unifyStructurally` change a type, so its
/// presence gates that step.
static bool carriesResolvableProjection(Type ty) {
  bool found = false;
  ty.walk([&](Type sub) -> WalkResult {
    if (auto proj = dyn_cast<ProjectionType>(sub);
        proj && !isPolymorphicType(proj)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Equate two types by equating their children: same constructor, same arity,
/// equal attributes, then unify corresponding child types. This is exact only
/// for injective constructors; a constructor that normalizes its arguments when
/// a type is built can make two types equal whose children are not, which
/// `unifyStructurally` reconciles before reaching here.
static LogicalResult unifyChildwise(Type formal,
                                    Type actual,
                                    ModuleOp module,
                                    UnificationMap &subst,
                                    llvm::function_ref<InFlightDiagnostic()> err) {
  if (formal == actual) return success();

  // check for same
  // 1. type constructor
  // 2. subelement arity
  // 3. attribute equality
  // and then recurse on children, if there are any
  auto [formalSubTys, formalSubAttrs] = getImmediateSubElements(formal);
  auto [actualSubTys, actualSubAttrs] = getImmediateSubElements(actual);

  bool formalHasSubs = !formalSubTys.empty() || !formalSubAttrs.empty();
  bool actualHasSubs = !actualSubTys.empty() || !actualSubAttrs.empty();

  // if neither side is decomposable, they're unequal leaves -> mismatch
  // if only one side is decomposable, constructors differ in structure -> mismatch
  if (!formalHasSubs || !actualHasSubs) {
    if (err) err() << "type mismatch: expected " << formal
                   << " but found " << actual;
    return failure();
  }

  // the constructor and arity of subelements of both types must match before recursing
  if (formal.getTypeID() != actual.getTypeID() ||
      formalSubTys.size() != actualSubTys.size() ||
      formalSubAttrs.size() != actualSubAttrs.size()) {
    if (err) err() << "type mismatch: expected " << formal
                   << " but found " << actual;
    return failure();
  }

  // The attributes of both types must match exactly before recursing on
  // child types. XXX: this treats attributes as opaque, so it will not find
  // and unify types stored inside type-bearing attributes.
  for (auto [f, a] : llvm::zip(formalSubAttrs, actualSubAttrs)) {
    if (f != a) {
      if (err) err() << "attribute mismatch: expected " << f
                     << " but found " << a;
      return failure();
    }
  }

  // Recurse on each sub type pair
  for (auto [f, a] : llvm::zip(formalSubTys, actualSubTys)) {
    if (failed(unify(f, a, module, subst, err)))
      return failure();
  }

  return success();
}

/// Unify two types that neither side drove through UnificationTypeInterface.
///
/// The child-wise decomposition in `unifyChildwise` assumes each type
/// constructor is injective. A constructor that normalizes its arguments when a
/// type is built is not injective: two types it makes equal can decompose into
/// children that are not, so equating the children misses the equality the
/// constructor establishes. When either side carries a ground projection the
/// module can resolve, resolve it and let each enclosing type rebuild through
/// that type's own constructor -- the same construction-time normalization then
/// applies to the resolved form -- and unify the rebuilt types.
///
/// The rebuild is attempted only when the direct decomposition cannot already
/// equate the two, so a decomposition that succeeds keeps its exact result and
/// its exact demand record. The failed decomposition runs on a saved
/// substitution with its diagnostic held back so it leaves no binding behind;
/// the resolution probe is an answer computed only to decide whether a rebuild
/// changes anything, so it runs as a cross-check and records nothing.
///
/// This terminates. resolveProjectionsByLookup returns a fixed point of its own
/// rewrite, so a rebuilt type carries no resolvable ground projection left for
/// this step to change; the re-unification either settles the two types or falls
/// to `unifyChildwise`, which recurses only on strictly smaller children.
static LogicalResult unifyStructurally(Type formal,
                                       Type actual,
                                       ModuleOp module,
                                       UnificationMap &subst,
                                       llvm::function_ref<InFlightDiagnostic()> err) {
  if (formal == actual) return success();

  // With no resolvable projection to rebuild, the decomposition is exact.
  if (!module || !(carriesResolvableProjection(formal) ||
                   carriesResolvableProjection(actual)))
    return unifyChildwise(formal, actual, module, subst, err);

  // Try the decomposition once, on a saved substitution and with the diagnostic
  // held back. A decomposition that succeeds is the answer and has recorded
  // exactly what a direct decomposition would; a failure must leave no binding
  // and no diagnostic behind so the rebuild below runs cleanly.
  UnificationMap saved = subst;
  if (succeeded(unifyChildwise(formal, actual, module, subst, /*err=*/{})))
    return success();
  subst = saved;

  Type resolvedFormal, resolvedActual;
  {
    DemandCrossCheckScope quiet;
    resolvedFormal = resolveProjectionsByLookup(
        formal, module, DemandOrigin::DeclarationMatch, LookupScope::Ground);
    resolvedActual = resolveProjectionsByLookup(
        actual, module, DemandOrigin::DeclarationMatch, LookupScope::Ground);
  }
  if (resolvedFormal != formal || resolvedActual != actual)
    return unify(resolvedFormal, resolvedActual, module, subst, err);

  // Nothing resolved: the decomposition's failure is the answer. Re-run it only
  // to surface the diagnostic; its recording repeats the held-back attempt
  // above, so keep it silent.
  DemandCrossCheckScope quiet;
  return unifyChildwise(formal, actual, module, subst, err);
}

/// Records a monomorphic projection the unifier let stand: it equated the two
/// sides, or bound a variable to the projection, without asking any impl what
/// the projection resolves to. These sit apart from the lookup's miss arms --
/// nothing here consulted the lookup at all.
///
/// The test is a root test on purpose. A projection nested inside two
/// aggregates the unifier found literally equal goes unobserved, because
/// walking every equality would put a type traversal on the unifier's hottest
/// path; such a projection is observed here anyway whenever the two sides are
/// not already equal, since the structural recursion then brings it to this
/// entry on its own.
///
/// The unifier's signature names no caller, so `module` classifies the demand:
/// the module-free comparator is what a verifier holding no module reaches,
/// while a caller carrying one is the stage or a committed-fact match inside a
/// verifier, which the stage's suspension brackets cover.
/// Unify a projection with another type by their spellings as written. Two
/// entries reach here: a module-free comparison (a verifier passes no module)
/// and a module-capable resolution (a pass or a committed-fact substitution
/// build passes the module).
///
///  - Projection vs projection: require the same symbolic projection head, then
///    recurse through trait application and associated-type arguments. This
///    allows nested projections to justify equivalent spellings.
///  - Projection vs a free inference variable it does not occur in: bind the
///    variable to the projection.
///  - Projection vs any other type: under the module-capable entry, resolve the
///    projection if a unique impl binds it and unify the result; under the
///    module-free entry, an unresolved crossing is a strict mismatch and is
///    rejected. Only the module-capable entry, on an irreducible crossing no
///    committed fact determines, tolerates it (see the residual note below).
///
/// Two spellings that agree here denote one type, since the head and every
/// argument agree. Two that disagree may still denote one type -- the caller
/// below settles that by normalizing both and asking again.
static LogicalResult unifyProjectionAsSpelled(
    ProjectionType self,
    Type other,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // Projection trait applications carry type arguments inside an attribute,
  // so structural attribute equality is too strict. Compare the symbolic
  // projection head, then recurse through the type arguments.
  if (auto otherProj = mlir::dyn_cast<ProjectionType>(other)) {
    auto formalApp = self.getTraitApplication();
    auto actualApp = otherProj.getTraitApplication();
    if (formalApp.getTraitName() != actualApp.getTraitName() ||
        self.getAssocName() != otherProj.getAssocName()) {
      if (err)
        err() << "projection mismatch: expected " << self << " but found "
              << otherProj;
      return failure();
    }

    if (failed(unifyTypeRange(formalApp.getTypeArgs(), actualApp.getTypeArgs(),
                              module, subst, err)))
      return failure();
    return unifyTypeRange(self.getAssocTypeArgs(), otherProj.getAssocTypeArgs(),
                          module, subst, err);
  }

  // projection vs non-projection.
  //
  // A projection is an opaque type function whose value is fixed only by claim
  // evidence, not by unification. Against a free inference variable that does
  // not occur inside this projection there is a sound choice -- bind the
  // variable to the projection -- so delegate to the variable's own unifier. A
  // variable that DOES occur inside is NOT bound here: a projection is a
  // resolvable function, so `V = proj<...V...>` is a forwarding equation (V is a
  // fixpoint of the resolution), not an infinite type, and must not trip the
  // variable unifier's occurs check. It falls to the module-free rejection or
  // the module-capable resolution below rather than binding.
  if (auto otherVar = mlir::dyn_cast<InferenceType>(other)) {
    bool occurs = false;
    Type(self).walk([&](Type t) {
      if (t == other) occurs = true;
    });
    if (!occurs)
      return otherVar.unify(self, module, subst, err);
  }

  // A projection all of whose arguments are concrete and whose trait application
  // a unique module-visible impl binds has one determined resolution. Binding a
  // variable mid-solve mints such ground projections (binding V:=i64 turns
  // proj<@Prod[V]> into the ground proj<@Prod[i64]>), so a caller carrying a
  // module -- a pass, or a committed-fact substitution build -- resolves them
  // here and unifies the resolved type against `other`, catching a real mismatch
  // against the resolved concrete spelling. A verifier compares spellings with
  // no module (the module-free comparator); an equality check performs no module
  // lookup, so this step is skipped and an unresolved crossing is a strict
  // mismatch below.
  if (isMonomorphicType(self) && module) {
    Type resolved = resolveProjectionsByLookup(
        self, module, DemandOrigin::DeclarationMatch, LookupScope::Ground);
    if (resolved != Type(self))
      return trait::unify(resolved, other, module, subst, err);
  }

  // The projection did not resolve and meets a rigid non-projection type. The
  // module-free comparator (a verifier) holds no evidence for the equality:
  // spellings must be identical after substitution, so reject the crossing.
  if (!module) {
    if (err)
      err() << "projection mismatch: expected " << self << " but found "
            << other;
    return failure();
  }

  // The module-capable entry reached an irreducible crossing that no committed
  // fact determines here and accepts it without a binding. The IR is
  // authoritative: a claim carries its own proof, and this comparison's caches
  // are acceleration, not the record. So a crossing accepted here whose equality
  // is real is witnessed elsewhere in the IR -- a coerce citing the equality, or
  // the proof on the claim -- and one whose equality is false is refused where
  // that evidence is consumed (a false equality's coerce fails the erase
  // barrier). The entry runs at pass time and inside verifiers on committed-fact
  // matches, so this acceptance is not pass-exclusive.
  return success();
}

/// Unify a projection type with another type.
///
/// A projection's spelling is not its identity: resolving it substitutes the
/// selected impl's associated-type binding, and an impl that forwards its
/// associated type through its own type parameter (`type Element = B::Element`)
/// binds a fresh projection, so one type is spelled `Ten[tuple<V>]::Element`
/// in the position that reaches it through the view and `Ten[V]::Element` in
/// the position that reaches it through the base. Comparing those two as
/// written finds a head mismatch, or -- when the heads agree -- recurses into
/// arguments that do not, and a comparer recursing into arguments equates
/// `tuple<V>` with `V`, which is an infinite type the occurs check refuses.
/// Neither answer is about the program; both are about the spellings.
///
/// So the comparison is made against normal forms: compare as written first,
/// and where that fails, normalize both sides to a fixed point and ask again.
/// Normalizing only after a failure keeps the answer the same and the lookups
/// off the path that already agreed -- an agreement on spellings is an
/// agreement on types.
///
/// This terminates. The normalization returns a fixed point of its own rewrite,
/// so the re-comparison it hands on carries no spelling left for a second
/// normalization to change: that entry finds both sides already normal and falls
/// through to the spelled comparison, which recurses only into strictly smaller
/// arguments.
LogicalResult ProjectionType::unify(
    Type other,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // The spelled comparison runs on a saved substitution with its diagnostic
  // held back: an attempt that succeeds is the answer and has recorded exactly
  // what a direct comparison would, while one that fails must leave no binding
  // and no diagnostic behind so the normalized comparison below runs cleanly.
  UnificationMap saved = subst;
  if (succeeded(unifyProjectionAsSpelled(*this, other, module, subst,
                                         /*err=*/{})))
    return success();
  subst = saved;

  // The module is what makes a normal form reachable: it holds the impls whose
  // bindings the resolution substitutes. A comparer holding none has no way to
  // tell a forwarding spelling from a different type, so it compares as written
  // and rejects, which is the module-free comparator's own strictness.
  if (module) {
    // The normalization is an answer computed only to decide whether the two
    // sides meet once resolved, so it runs as a cross-check and records nothing.
    Type normalizedSelf, normalizedOther;
    {
      DemandCrossCheckScope quiet;
      normalizedSelf = resolveProjectionsByLookup(
          *this, module, DemandOrigin::DeclarationMatch, LookupScope::Determined);
      normalizedOther = resolveProjectionsByLookup(
          other, module, DemandOrigin::DeclarationMatch, LookupScope::Determined);
    }
    if (normalizedSelf != Type(*this) || normalizedOther != other)
      return trait::unify(normalizedSelf, normalizedOther, module, subst, err);
  }

  // Normalizing changed nothing, so the spelled comparison's failure is the
  // answer. Re-run it only to surface the diagnostic; its recording repeats the
  // held-back attempt above, so keep it silent.
  DemandCrossCheckScope quiet;
  return unifyProjectionAsSpelled(*this, other, module, subst, err);
}

/// Attempt to unify `formal` with `actual`, extending `subst` with any
/// new bindings that make them equal under substitution.
///
/// Both sides are first normalized by applying `subst` to a fixed point.
/// After that we check for trivial equality and then choose how to drive
/// unification:
///
/// Priority of unifiers:
///  1. **Formal first** — If the formal side implements
///     `UnificationTypeInterface`, we let it drive unification. This gives
///     formal-side types (inference variables, projections, claims) first
///     refusal to decide how to handle the match.
///  2. **Actual second** — If the actual side implements
///     `UnificationTypeInterface`, we let it drive. This handles the
///     symmetric case (e.g., inference variable on the actual side).
///  3. **Structural fallback** — Otherwise we fall back to generic
///     shape-by-shape unification for non-unifiable types.
///
/// Returns success if the two types can be made equal under an extended `subst`.
/// On failure, nothing is recorded and `err` (if provided) will be invoked to
/// emit a diagnostic.
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize both types by applying the current substitution
  formal = applySubstitutionToFixedPoint(subst.toTypeMap(), formal);
  actual = applySubstitutionToFixedPoint(subst.toTypeMap(), actual);

  // if the normalized types are equal, unification succeeds
  if (formal == actual)
    return success();

  // formal-side unifier takes priority
  if (auto formalUnifier = dyn_cast<UnificationTypeInterface>(formal))
    return formalUnifier.unify(actual, module, subst, err);

  // actual-side unifier
  if (auto actualUnifier = dyn_cast<UnificationTypeInterface>(actual))
    return actualUnifier.unify(formal, module, subst, err);

  // structural fallback
  return unifyStructurally(formal, actual, module, subst, err);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst) {
  auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
  return unify(formal, actual, module, subst, errFn);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  UnificationMap discardedSubst;
  return unify(formal, actual, module, discardedSubst, err);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module) {
  UnificationMap discardedSubst;
  return unify(formal, actual, module, discardedSubst);
}


//===----------------------------------------------------------------------===//
// instantiate
//===----------------------------------------------------------------------===//

Type instantiate(Type root, InstantiationMap &inst, uint64_t &idCounter) {
  AttrTypeReplacer r = makeEndpointSealedReplacer();
  r.addReplacement([&](Type t) -> std::optional<Type> {
    if (auto generic = dyn_cast<GenericTypeInterface>(t)) {
      return generic.instantiate(inst, idCounter);
    }
    return std::nullopt;
  });

  // Instantiate the equality endpoints the seal holds as a leaf: otherwise a
  // formal claim<!poly = T> would keep a rigid poly and never share the
  // inference variable the rest of the formal instantiates to, so a claim
  // endpoint variable unifies across a call boundary like any other.
  r.addReplacement(
      [&](ClaimType claim) -> std::optional<std::pair<Type, WalkResult>> {
    return respellEqualityEndpoints(claim, [&](Type t) {
      return instantiate(t, inst, idCounter);
    });
  });

  // this walks into types nested inside attributes (e.g., trait applications)
  // and replaces all GenericTypeInterface types according to (and extending) inst
  return r.replace(root);
}


/// The first inference id no variable spelled in `types` already uses.
///
/// An inference variable's identity is its id, so a mint that starts over at
/// zero hands back a variable a type in hand may already spell -- and the two,
/// being one type, then unify as one variable. That happens whenever a
/// specialization is built over types an enclosing specialization already
/// instantiated: this build's first fresh variable would alias the enclosing
/// build's first. Starting past every id in hand is what makes fresh mean fresh.
///
/// An equality claim's endpoints are ordinary sub-elements, so the structural
/// walk reaches a variable spelled only inside one -- the same universe
/// respellEqualityEndpoints rewrites. Otherwise the mint would start past every
/// id but those, and a fresh variable would alias one an endpoint holds.
static uint64_t firstUnusedInferenceId(ArrayRef<Type> types) {
  uint64_t next = 0;
  for (Type ty : types)
    ty.walk([&](Type sub) {
      if (auto var = dyn_cast<InferenceType>(sub))
        next = std::max(next, var.getUniqueId() + 1);
      return WalkResult::advance();
    });
  return next;
}

FailureOr<SpecializationMap> buildSpecialization(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // instantiate generics on both sides with the same instantiation map
  InstantiationMap genToInfer;
  uint64_t idCounter = firstUnusedInferenceId({formal, actual});
  Type iformal = instantiate(formal, genToInfer, idCounter);
  Type iactual = instantiate(actual, genToInfer, idCounter);

  // get the inverse instantiation map as well
  auto inferToGen = invertSubstitution(genToInfer.toTypeMap(), err);
  if (failed(inferToGen)) return failure();

  // unify the instantiated formal and actual types
  UnificationMap inferToType;
  if (failed(unify(iformal, iactual, module, inferToType, err)))
    return failure();

  // compose (gen -> infer) o (infer -> type)
  auto composed = composeSubstitutions(genToInfer.toTypeMap(), inferToType.toTypeMap(), err);
  if (failed(composed)) return failure();

  // compose again with inferToGen to map any remaining unsolved
  // inference variables originating from actual back to their
  // original generics
  auto result = composeSubstitutions(*composed, *inferToGen, err);
  if (failed(result)) return failure();

  normalizeSubstitutionInPlace(*result);
  return SpecializationMap::fromTypeMap(*result);
}


//===----------------------------------------------------------------------===//
// Declaration matching
//===----------------------------------------------------------------------===//

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
