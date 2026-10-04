// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <cstdint>
#include <string>
#include <llvm/ADT/ScopeExit.h>
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

/// The sealed replacer plus one projection rule, over the spellings `reads`
/// admits.
static AttrTypeReplacer makeProjectionReplacer(
    bool (*reads)(ProjectionType),
    std::function<std::optional<Type>(ProjectionType)> hop) {
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement(
      [reads, hop = std::move(hop)](Type t) -> std::optional<Type> {
        auto projection = dyn_cast<ProjectionType>(t);
        if (!projection || !reads(projection))
          return std::nullopt;
        return hop(projection);
      });
  return replacer;
}

AttrTypeReplacer makeGroundProjectionReplacer(
    std::function<std::optional<Type>(ProjectionType)> hop) {
  return makeProjectionReplacer(
      [](ProjectionType proj) { return !isPolymorphicType(Type(proj)); },
      std::move(hop));
}

AttrTypeReplacer makeGroundHeadProjectionReplacer(
    std::function<std::optional<Type>(ProjectionType)> hop) {
  return makeProjectionReplacer(
      [](ProjectionType proj) {
        return !isPolymorphicType(Type(proj.asClaim()));
      },
      std::move(hop));
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
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams) {
  SmallVector<Type> concreteTypes;
  for (auto ty : typeParams)
    concreteTypes.push_back(subst.apply(ty));
  return generateMangledNameSuffixFor(concreteTypes);
}


//===----------------------------------------------------------------------===//
// Ground projection resolution
//===----------------------------------------------------------------------===//

LogicalResult checkObligationChainDepth(ArrayRef<ObligationFrame> chain,
                                        unsigned height) {
  return success(chain.size() + height - 1 < kInstantiationDepthLimit);
}

void emitObligationOverflow(Location anchor, TraitApplicationAttr app,
                            ArrayRef<ObligationFrame> chain, unsigned height) {
  size_t depth = chain.size() + height - 1;
  InFlightDiagnostic diagnostic =
      emitError(anchor)
      << "overflow evaluating the requirement '" << app << "': " << depth
      << " obligations stand on the chain that reaches it";
  nameChainEnds<ObligationFrame>(
      diagnostic, chain, [](InFlightDiagnostic &d, ObligationFrame frame) {
        Diagnostic &note = d.attachNote();
        note << "required by " << frame.application;
        if (frame.proof)
          note << ", stated by proof " << frame.proof;
      });
}

LogicalResult tryNormalizeProjectionsToFixedPoint(
    Type ty, llvm::function_ref<Type(Type)> step, Type &out) {
  Type previous;
  for (unsigned i = 0; i != kInstantiationDepthLimit && ty != previous; ++i) {
    previous = ty;
    ty = step(ty);
  }
  // `out` carries what the loop reached either way: the fixed point on success,
  // the still-changing partial normal form on failure. A caller reporting the
  // nonconvergence reads the partial to name the type that would not settle.
  out = ty;
  return success(ty == previous);
}

// The body of `resolveProjectionsByLookup`. `candidates` holds, for one
// reading, the candidate impls each application's lookup has found, which the
// reading and the header readings under it share. `converged` reports whether
// the fixed-point driver reached a normal form: on false, `ty` carries the
// driver's partial (the still-unresolved projection spelled as written), which
// the entry names as it refuses.
static Type resolveProjectionsByLookupCore(
    Type ty, ModuleOp module, LookupScope scope,
    DenseMap<TraitApplicationAttr, SmallVector<ImplOp>> &candidates,
    bool &converged) {
  converged = true;
  if (!module)
    return ty;

  // The context a candidate's header is read through here: this lookup itself,
  // at the same scope, so a header spelling a
  // projection (`impl<T> Index<T::Shape, T::Element> for T`) reproduces a demand
  // spelling the resolution and is read by the rule the demand is read by. A
  // reading that does not converge is a header this reading cannot rebuild.
  auto byLookup = [&](Type ty) -> FailureOr<Type> {
    bool headerConverged;
    Type read = resolveProjectionsByLookupCore(ty, module, scope, candidates,
                                               headerConverged);
    if (!headerConverged)
      return failure();
    return read;
  };

  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
    // Which impl serves a projection is decided by its head application alone:
    // the associated-type binding the impl states is a function of the
    // projection's own associated-type arguments, so a head naming one impl
    // answers whatever those arguments still spell. A head still carrying a
    // variable stands for as many impls as that variable has instances, and no
    // one impl answers for it.
    const bool polymorphicHead = isPolymorphicType(Type(proj.asClaim()));

    // A projection whose head still carries variables resolves only under the
    // determined scope, and then only if its own spelling picks the impl (the
    // one-way match below).
    if (polymorphicHead && scope == LookupScope::Ground)
      return std::nullopt;

    ClaimType claim = proj.asClaim();
    TraitApplicationAttr app = claim.getTraitApplication();

    // Read-only selection: resolve only when exactly one existing impl binds
    // this application. Two or more matches, and impl generation, are left to
    // the resolver. The single match may be conditional (a nonempty assumptions
    // list): selecting it is mechanical name resolution, not premise evaluation,
    // and a legal program has already discharged this ground projection's head
    // claim -- the premise the conditional impl carries.
    //
    // The candidates are read once per application and reading, since this
    // lookup mutates no impl. An application whose candidates are being read
    // stands with none until they are read: a candidate header that reads the
    // same application again -- an impl whose header projects through its own
    // trait at the application asked -- finds no candidate there, as a cycle
    // guard refuses an application it meets again.
    auto it = candidates.find(app);
    if (it == candidates.end()) {
      auto trait = app.getTrait(module, nullptr);
      if (failed(trait))
        return std::nullopt;
      candidates.try_emplace(app);
      SmallVector<ImplOp> found = trait->getCandidateImplsFor(claim, byLookup);
      it = candidates.find(app);
      it->second = std::move(found);
    }
    if (it->second.size() != 1)
      return std::nullopt;
    ImplOp impl = it->second.front();

    // A projection over a type variable in its head denotes one type at every
    // instance of that variable, so the impl serving it must serve every
    // instance: it may carry no premise. An impl with a where clause serves the
    // instances its premises admit and no others, and which those are is settled
    // per instance, so it answers for none of them here. The head claim that
    // licenses reading such an impl for a ground head is discharged at an
    // instance, not at this spelling.
    if (polymorphicHead && !impl.getWhereClaims().empty())
      return std::nullopt;

    // Nothing the projection spells is narrowed to fit the impl: the impl's own
    // parameters take the arguments standing opposite them and the header
    // rebuilt at those must be the projection's application. So an impl the
    // projection could only reach by narrowing one of its variables is refused
    // here, and no separate one-way test stands over this one.
    auto subst = impl.buildSubstitutionForSelfClaim(claim, byLookup,
                                                    /*errFn=*/nullptr);
    if (failed(subst))
      return std::nullopt;

    auto binding = impl.specializeAssociatedTypeBinding(
        proj.getAssocName().getValue(), proj.getAssocTypeArgs(), *subst);
    if (failed(binding))
      return std::nullopt;

    return *binding;
  });

  // A resolved binding may itself expose a ground projection, so run to a
  // fixed point. A chain that never grounds leaves `ty` at the driver's partial
  // normal form and reports nonconvergence to the caller.
  converged = succeeded(tryNormalizeProjectionsToFixedPoint(
      ty, [&](Type t) { return replacer.replace(t); }, ty));
  return ty;
}

FailureOr<Type> resolveProjectionsByLookup(
    Type ty, ModuleOp module, LookupScope scope,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  DenseMap<TraitApplicationAttr, SmallVector<ImplOp>> candidates;
  bool converged;
  Type out =
      resolveProjectionsByLookupCore(ty, module, scope, candidates, converged);
  // The fallible entry refuses a projection that will not ground so a verifier
  // reached from untrusted IR fails cleanly rather than admitting the cycle.
  if (!converged) {
    if (emitError)
      emitError() << "projection normalization did not converge within "
                  << kInstantiationDepthLimit
                  << " projection steps for type " << out;
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
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

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

Citation verifyCitation(ClaimType unproven, ClaimType proven, ModuleOp module,
                        Normalizer normalize,
                        llvm::function_ref<InFlightDiagnostic()> err) {
  // inspect the proof symbol on the proven side
  auto symOp = ProofOp::getProofOpOrUnconditionalImplOp(module, proven.getProof(), err);
  if (failed(symOp)) return Citation::Refused;

  // The declaration the cited symbol holds: an unconditional impl's header, or
  // the claim a proof derives. A proof op may be written over type variables
  // and stand for every instance of them, so its parameters take the arguments
  // the obligation supplies and the claim it rebuilds must be that obligation.
  // What a proof proves underneath was decided at its own body, so nothing here
  // reads inside it.
  Type declaration = isa<ImplOp>(*symOp)
                         ? Type(cast<ImplOp>(*symOp).getSelfClaim())
                         : Type(cast<ProofOp>(*symOp).getProvenClaim().asUnproven());

  // A side still spelling a projection once the caller's evidence has been read
  // is one nothing here can decide: impl selection resolves it through the
  // candidate it settles on, which a reader holding no record cannot. Such an
  // obligation is declined rather than discharged, so the claim stands unproven
  // for selection to derive and for the leftover walk to refuse.
  if (succeeded(matchDeclaration(getTypeParametersIn(declaration), declaration,
                                 Type(unproven), normalize, /*err=*/nullptr)))
    return Citation::Carries;

  FailureOr<Type> readObligation = normalize(Type(unproven));
  FailureOr<Type> readDeclared = normalize(declaration);
  if (failed(readObligation) || failed(readDeclared) ||
      containsType<ProjectionType>(*readObligation) ||
      containsType<ProjectionType>(*readDeclared))
    return Citation::Declined;

  if (err) err() << "proof " << proven.getProof() << " proves " << declaration
                 << ", which does not discharge the obligation "
                 << *readObligation;
  return Citation::Refused;
}

void walkObligationSites(Type root, llvm::function_ref<void(Type)> visit) {
  root.walk<WalkOrder::PreOrder>([&](Type sub) -> WalkResult {
    visit(sub);
    if (auto claim = dyn_cast<ClaimType>(sub))
      if (claim.isEquality())
        return WalkResult::skip();
    return WalkResult::advance();
  });
}

bool isUndischargedObligation(Type site) {
  if (auto claim = dyn_cast<ClaimType>(site))
    return claim.isApplication() && claim.isMonomorphic() && !claim.isProven();
  return isa<ProjectionType>(site) && !isPolymorphicType(site);
}

bool carriesUndischargedObligation(Type root) {
  bool found = false;
  walkObligationSites(root, [&](Type site) {
    found = found || isUndischargedObligation(site);
  });
  return found;
}

LogicalResult verifyCitationsIn(Type ty, ModuleOp module, Normalizer normalize,
                                llvm::function_ref<InFlightDiagnostic()> err) {
  LogicalResult status = success();

  ty.walk([&](Type node) {
    if (status.failed()) return;

    auto claim = dyn_cast<ClaimType>(node);
    if (!claim || !claim.isProven())
      return;

    if (verifyCitation(claim.asUnproven(), claim, module, normalize, err) ==
        Citation::Refused)
      status = failure();
  });

  return status;
}

namespace {

/// The evidence one proven claim stands on: the impl its proof derives it from
/// -- or the unconditional impl it names directly -- the arguments that impl's
/// parameters take at the claim, and the claims the proof's derive supplies its
/// where entries, carried to the claim.
struct CitedEvidence {
  ImplOp impl;
  SpecializationMap arguments;
  SmallVector<ClaimType> premises;
};

} // namespace

/// Reads the evidence `claim`'s proof stands on, failing when a symbol the
/// claim names is absent.
static FailureOr<CitedEvidence> readCitedEvidence(
    ClaimType claim, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto cited =
      ProofOp::getProofOpOrUnconditionalImplOp(module, claim.getProof(), errFn);
  if (failed(cited))
    return failure();
  CitedEvidence evidence;
  auto proof = dyn_cast<ProofOp>(*cited);
  if (!proof) {
    evidence.impl = cast<ImplOp>(*cited);
    return evidence;
  }
  evidence.impl = proof.getImpl();
  if (!evidence.impl) {
    if (errFn)
      errFn() << "proof '" << claim.getProof() << "' cites no impl";
    return failure();
  }
  auto arguments = proof.getImplArgumentsAt(claim, errFn);
  auto premises = proof.getPremisesAt(claim, errFn);
  if (failed(arguments) || failed(premises))
    return failure();
  evidence.arguments = std::move(*arguments);
  evidence.premises = std::move(*premises);
  return evidence;
}

FailureOr<uint64_t> getClaimRequirementCount(
    ClaimType claim,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // An equality claim states that two spellings name one type; it applies no
  // trait, so it requires nothing.
  if (claim.isEquality())
    return 0;

  auto trait = claim.getTraitApplication().getTrait(module, errFn);
  if (failed(trait))
    return failure();
  uint64_t count = trait->getRequirements().size();
  if (!claim.isProven())
    return count;
  auto cited =
      ProofOp::getProofOpOrUnconditionalImplOp(module, claim.getProof(), errFn);
  if (failed(cited))
    return failure();
  if (auto proof = dyn_cast<ProofOp>(*cited))
    count += proof.getDerive().getAssumptions().size();
  return count;
}

FailureOr<ClaimType> getClaimRequirementAt(
    ClaimType claim,
    ModuleOp module,
    uint64_t index,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto count = getClaimRequirementCount(claim, module, errFn);
  if (failed(count))
    return failure();
  if (index >= *count) {
    if (errFn)
      errFn() << "requirement index " << index << " is out of range: " << claim
              << " has " << *count << " requirements";
    return failure();
  }

  TraitOp trait = claim.getTraitApplication().getTraitOrAbort(
      module, "getClaimRequirementAt: the counted trait vanished");
  uint64_t traitCount = trait.getRequirements().size();

  // An unproven claim carries no evidence, so each requirement it reaches is
  // the trait's at its arguments, unproven.
  if (!claim.isProven())
    return trait.specializeRequirementAsClaimFor(claim, index, errFn);

  auto evidence = readCitedEvidence(claim, module, errFn);
  if (failed(evidence))
    return failure();

  // A where entry of the impl: the entry at the impl's arguments, carrying the
  // proof the citation supplies there.
  if (index >= traitCount) {
    unsigned position = index - traitCount;
    auto entry = cast<ClaimType>(instantiate(
        Type(evidence->impl.getWhereClaims()[position]), evidence->arguments));
    ClaimType carrying = entry.carryingProofOf(evidence->premises[position]);
    return carrying ? carrying : entry;
  }

  // A requirement of the trait: the requirement at the claim's arguments, read
  // through the impl's own binding as the impl's return is spelled. The
  // evidence is the impl's return operand there, which the stage inlines where
  // a projection reads it (`ProjectOp::inlineEvidence`); this names the claim
  // alone, unproven.
  auto requirement =
      trait.specializeRequirementAsClaimFor(claim.asUnproven(), index, errFn);
  if (failed(requirement))
    return failure();
  ClaimType obligation = *requirement;
  if (obligation.isEquality())
    return obligation;
  NormalizationContext own;
  own.addLocalProjectionRule(evidence->impl,
                             claim.asUnproven().getTraitApplication(),
                             evidence->arguments);
  auto read = own.normalize(Type(obligation), errFn);
  if (failed(read))
    return failure();
  return cast<ClaimType>(*read);
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
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

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
