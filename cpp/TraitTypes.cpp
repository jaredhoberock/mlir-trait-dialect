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
  if (a != b)
    parent[b] = a;
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

LogicalResult verifyCitation(ClaimType proven, ModuleOp module,
                             llvm::function_ref<InFlightDiagnostic()> err) {
  auto symOp =
      ProofOp::getProofOpOrUnconditionalImplOp(module, proven.getProof(), err);
  if (failed(symOp))
    return failure();
  TraitApplicationAttr declared =
      isa<ImplOp>(*symOp) ? cast<ImplOp>(*symOp).getSelfApplication()
                          : cast<ProofOp>(*symOp).getTraitApplication();
  // A claim names the proof of exactly its own spelling: every writer mints a
  // proof at the application the claim it proves spells, so the citation is
  // one comparison.
  if (declared == proven.getTraitApplication())
    return success();
  if (err)
    err() << "proof " << proven.getProof() << " proves "
          << Type(ClaimType::get(proven.getContext(), declared))
          << ", which does not discharge the obligation "
          << Type(proven.asUnproven());
  return failure();
}

void walkObligationSites(Type root, llvm::function_ref<void(Type)> visit) {
  root.walk<WalkOrder::PreOrder>([&](Type sub) -> WalkResult {
    visit(sub);
    return isa<ClaimType>(sub) ? WalkResult::skip() : WalkResult::advance();
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

  // A requirement of the trait -- every requirement an unproven claim
  // reaches -- is the trait's at the claim's arguments, as the claim spells
  // them, unproven. Of a proven claim, the evidence is the impl's return
  // operand there, which the stage inlines where a projection reads it
  // (`ProjectOp::inlineEvidence`); this names the claim alone.
  if (index < traitCount)
    return trait.specializeRequirementAsClaimFor(claim.asUnproven(), index,
                                                 errFn);

  // A where entry of the impl: the claim the proof's derive is given at that
  // position, read by index. Only a proof has where entries to read: an impl
  // named directly is unconditional.
  auto cited =
      ProofOp::getProofOpOrUnconditionalImplOp(module, claim.getProof(), errFn);
  if (failed(cited))
    return failure();
  return cast<ClaimType>(cast<ProofOp>(*cited)
                             .getDerive()
                             .getAssumptions()[index - traitCount]
                             .getType());
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
  if (rebuilt == wanted)
    return success();
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
