#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include "TraitTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TraitTypes.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TraitTypes.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

int freshPolyTypeId() {
  static std::atomic<int> counter{-1};
  return counter.fetch_sub(1, std::memory_order_relaxed);
}

PolyType PolyType::fresh(MLIRContext* ctx) {
  return PolyType::get(ctx, freshPolyTypeId());
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int uniqueId = 0;

  // parse this:
  // <fresh> or
  // <int>

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  if (succeeded(parser.parseOptionalKeyword("fresh"))) {
    uniqueId = freshPolyTypeId();
  } else {
    if (parser.parseInteger(uniqueId)) {
      parser.emitError(parser.getNameLoc(), "expected integer or 'fresh'");
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

LogicalResult PolyType::substituteWith(
  Type other,
  ModuleOp /*module*/,
  DenseMap<Type,Type> &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;
  
  // normalize
  other = applySubstitution(subst, other);

  // first check for equality
  if (self == other)
    return success();

  // If we've already recorded a substitution for self, check consistency.
  if (auto it = subst.find(self); it != subst.end()) {
    if (it->second != other) {
      if (err)
        return err() << "mismatched substitution for type "
                     << self << ": expected "
                     << it->second << ", but found " << other;
      return failure();
    }
    return success();
  }

  // check for recursive substitution
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
  subst[self] = other;
  return success();
}


//===----------------------------------------------------------------------===//
// ClaimType
//===----------------------------------------------------------------------===//

LogicalResult ClaimType::verify(function_ref<InFlightDiagnostic()> emitError,
                                TraitApplicationAttr app,
                                FlatSymbolRefAttr proof) {
  auto fail = [&]() -> LogicalResult {
    return emitError ? emitError() << "nested !trait.claim types are not allowed"
                     : failure();
  };

  for (Type t : app.getTypeArgs()) {
    bool nested = false;
    t.walk([&](Type sub) {
      if (mlir::isa<ClaimType>(sub))
        nested = true;
    });
    if (nested)
      return fail();
  }

  bool polymorphic = llvm::any_of(app.getTypeArgs(), [](Type ty) {
    return isPolymorphicType(ty);
  });

  if (polymorphic && proof) {
    if (emitError) emitError() << "A polymorphic !trait.claim cannot have a proof";
    return failure();
  }

  return success();
}

LogicalResult ClaimType::verifySymbolUses(ModuleOp module, llvm::function_ref<InFlightDiagnostic()> err) {
  // verify trait application
  if (failed(getTraitApplication().verifyTraitApplication(module, err)))
    return failure();

  // if there's a proof, verify that it points to a valid symbol
  if (auto proof = getProof()) {
    if (failed(ProofOp::getProofOpOrSelfProofImplOp(module, proof, err)))
      return failure();
  }

  return success();
}

Type ClaimType::parse(AsmParser& p) {
  MLIRContext *ctx = p.getContext();

  if (p.parseLess())
    return {};

  TraitApplicationAttr app = mlir::dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app)
    return {};

  FlatSymbolRefAttr proof;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(proof))
      return {};
  }

  if (p.parseGreater())
    return {};

  auto emitError = [&] { return p.emitError(p.getCurrentLocation()); };
  return ClaimType::getChecked(emitError, ctx, app, proof);
}

void ClaimType::print(AsmPrinter& p) const {
  p << "<";
  getTraitApplication().print(p);
  if (isProven()) {
    p << " by " << getProof();
  }
  p << ">";
}

bool ClaimType::isMonomorphic() const {
  // a !trait.claim<@Trait[Types...]> is monomorphic is all of its type arguments are monomorphic
  return llvm::all_of(getTraitApplication().getTypeArgs(), [](Type ty) {
    return mlir::trait::isMonomorphicType(ty);
  });
}

static LogicalResult verifyAndRecordProvenClaim(
    ClaimType unproven,
    ClaimType proven,
    ModuleOp module,
    DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // early exit if we've already recorded this proof
  if (auto it = subst.find(unproven); it != subst.end()) {
    if (it->second != proven) {
      if (err) err() << "inconsistent proof mapping: " << unproven
                     << " is already bound to " << it->second
                     << ", but attempted to bind " << proven;
      return failure();
    }
    return success();
  }

  // verify preconditions
  if (unproven.isProven()) { if (err) err() << "expected unproven claim"; return failure(); }
  if (!proven.isProven()) { if (err) err() << "expected proven claim"; return failure(); }

  // look up the trait and its requirements using the unproven claim
  auto trait = unproven.getTraitApplication().getTrait(module, err);
  if (failed(trait)) return failure();

  // get trait requirements
  SmallVector<ClaimType> requirements = trait->getRequirementsAsClaimsWith(subst);

  // inspect the proof symbol on the proven side
  auto symOp = ProofOp::getProofOpOrSelfProofImplOp(module, proven.getProof(), err);
  if (failed(symOp))
    return failure();

  // if it's an impl op, check that we have no requirements
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (!requirements.empty()) {
      if (err) err() << "self-proving impl provides no subproof for trait requirements";
      return failure();
    }

    // success: bind the whole claim so that later normalization keeps the proof
    subst[unproven] = proven;
    return success();
  }

  // otherwise it must be a ProofOp
  auto proof = dyn_cast<ProofOp>(*symOp);

  // check that the proof's claim can substitute with proven
  if (failed(proof.getProvenClaim().substituteWith(proven, module, subst, err))) 
    return failure();

  // get impl assumptions from the proven side
  SmallVector<ClaimType> assumptions = proof.getImpl().getAssumptionsAsClaimsWith(subst);

  // concatenate requirements + assumptions into obligations
  SmallVector<ClaimType> obligations = std::move(requirements);
  obligations.append(assumptions);

  // get the subproof claims
  auto subproofs = proof.verifyAndGetSubproofClaims(err);
  if (failed(subproofs)) return failure();

  // the number of subproofs must match the number of obligations 
  if (subproofs->size() != obligations.size()) {
    if (err) err() << "arity mismatch: expected " << obligations.size()
                   << ", but found " << subproofs->size();
    return failure();
  }

  // recurse over obligations
  for (auto [ob, sub] : llvm::zip(obligations, *subproofs)) {
    if (failed(verifyAndRecordProvenClaim(ob, sub, module, subst, err)))
      return failure();
  }

  // success: bind the whole claim so that later normalization keeps the proof
  subst[unproven] = proven;
  return success();
}

void ClaimType::getProjections(
    ModuleOp module,
    SmallVectorImpl<ClaimType>& result,
    ClaimType::ProjectionKind kinds) {
  // identity?
  if (kinds & Identity) {
    result.push_back(*this);
  }

  // trait requirements?
  if (kinds & TraitRequirements) {
    auto trait = getTraitApplication().getTraitOrAbort(module, "ClaimType::getProjections: couldn't find trait");
    auto subst = trait.buildSubstitutionFor(*this);
    result.append(trait.getRequirementsAsClaimsWith(subst));
  }

  // proven impl assumptions?
  if (kinds & ProvenImplAssumptions and isProven()) {
    // proven impl assumptions exist only if the proof is a reference to a ProofOp
    // self-proving ImplOps have no assumptions
    if (auto proof = SymbolTable::lookupNearestSymbolFrom<ProofOp>(module, getProof())) {
      result.append(proof.getImplAssumptionClaims());
    }
  }
}

LogicalResult ClaimType::substituteWith(
    Type other,
    ModuleOp module,
    DenseMap<Type,Type>& subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize formal first
  Type formalNormTy = applySubstitution(subst, *this);
  ClaimType formal = mlir::dyn_cast<ClaimType>(formalNormTy);

  // if formal is no longer a ClaimType, delegate to generic path
  if (!formal)
    return trait::substituteWith(formalNormTy, other, module, subst, err);

  // normalize actual second
  Type normActualTy = applySubstitution(subst, other);
  ClaimType actual = mlir::dyn_cast<ClaimType>(normActualTy);

  // if actual isn't a claim, it's an immediate mismatch
  if (!actual) {
    if (err) {
      err() << "expected !trait.claim, but found " << normActualTy;
    }
    return failure();
  }

  // do claim-specific checks below

  auto formalApp = formal.getTraitApplication();
  auto actualApp = actual.getTraitApplication();

  // same trait?
  if (formalApp.getTraitName() != actualApp.getTraitName()) {
    if (err) err() << "trait mismatch: expected " << formalApp.getTraitName()
                   << ", but found " << actualApp.getTraitName();
    return failure();
  }

  // same arity?
  auto formalArgs = formalApp.getTypeArgs();
  auto actualArgs = actualApp.getTypeArgs();
  if (formalArgs.size() != actualArgs.size()) {
    if (err) err() << "arity mismatch: expected " << formalArgs.size()
                   << " type arguments, but found " << actualArgs.size();
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
    if (err) err() << "cannot substitute proven claim with unproven claim";
    return failure();
  }

  // recurse on each argument pair
  for (auto [f, a] : llvm::zip(formalArgs, actualArgs)) {
    if (failed(trait::substituteWith(f, a, module, subst, err)))
      return failure();
  }

  // recursively record whole-claim substitutions when:
  // 1. formal is unproven, and
  // 2. actual is proven
  // this ensures that all proven claims resulting from substitution application are always associated with proofs
  if (!formal.isProven() && actual.isProven()) {
    if (failed(verifyAndRecordProvenClaim(formal, actual, module, subst, err)))
      return failure();
  }

  return success();
}


//===----------------------------------------------------------------------===//
// substituteWith
//===----------------------------------------------------------------------===//

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

LogicalResult substituteWith(Type formal,
                             Type actual,
                             ModuleOp module,
                             llvm::DenseMap<Type, Type> &subst,
                             llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize both types by applying the current substitution
  formal = applySubstitution(subst, formal);
  actual = applySubstitution(subst, actual);

  // give the formal type first right of refusal
  if (auto mti = dyn_cast<MonomorphizableTypeInterface>(formal))
    return mti.substituteWith(actual, module, subst, err);

  // if the normalized types are equal, unification succeeds
  if (formal == actual)
    return success();

  // structural fallback: check for same
  // 1. type constructor
  // 2. subelement arity
  // 3. attribute equality
  // and then recurse on children, if there are any
  auto [formalSubTys, formalSubAttrs] = getImmediateSubElements(formal);
  auto [actualSubTys, actualSubAttrs] = getImmediateSubElements(actual);

  bool formalHasSubs = !formalSubTys.empty() || !formalSubAttrs.empty();
  bool actualHasSubs = !actualSubTys.empty() || !actualSubTys.empty();

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

  // the attributes of both types must match exactly before recursing on child types
  for (auto [f, a] : llvm::zip(formalSubAttrs, actualSubAttrs)) {
    llvm::errs() << "comparing " << f << " and " << a << "\n";
    if (f != a) {
      if (err) err() << "attribute mismatch: expected " << f
                     << " but found " << a;
      return failure();
    }
  }

  // Recurse on each sub type pair
  for (auto [f, a] : llvm::zip(formalSubTys, actualSubTys)) {
    if (failed(substituteWith(f, a, module, subst, err)))
      return failure();
  }

  return success();
}

LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::DenseMap<Type,Type> &subst) {
  auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
  return substituteWith(formal, actual, module, subst, errFn);
}

LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DenseMap<Type,Type> discardedSubst;
  return substituteWith(formal, actual, module, discardedSubst, err);
}

LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module) {
  DenseMap<Type,Type> discardedSubst;
  return substituteWith(formal, actual, module, discardedSubst);
}


} // end mlir::trait
