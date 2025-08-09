#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include "TypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}

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


LogicalResult PolyType::unifyWith(
  Type other,
  ModuleOp module,
  DenseMap<Type,Type> &substitution,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;

  // If we've already substituted a concrete for self, check consistency.
  if (auto it = substitution.find(self); it != substitution.end()) {
    if (it->second != other) {
      if (err)
        return err() << "mismatched substitution for type "
                     << self << ": expected "
                     << it->second << ", but found " << other;

      return failure();
    }
    return success();
  }

  // XXX TODO do an occurs check

  // bind the variable
  substitution[self] = other;
  return success();
}


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
  return llvm::all_of(getTypeArgs(), [](Type ty) {
    return mlir::trait::isMonomorphicType(ty);
  });
}

static SmallVector<ClaimType> getRequirements(ClaimType claimTy, ModuleOp module) {
  // lookup the TraitOp
  auto traitOp = SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, claimTy.getTraitApplication().getTrait());
  if (!traitOp)
    llvm::report_fatal_error("getRequirements: unable to resolve TraitOp");

  auto subst = traitOp.buildSubstitutionFor(claimTy);

  // apply substitution to each polymorphic trait requirement
  SmallVector<ClaimType> result;
  for (ClaimType polyClaim : traitOp.getRequirementsAsClaims()) {
    ClaimType substClaim = dyn_cast<ClaimType>(applySubstitution(subst, polyClaim));
    if (!substClaim)
      llvm::report_fatal_error("getRequirements: expected ClaimType");

    result.push_back(substClaim);
  }

  return result;
}

static SmallVector<ClaimType> getSubproofs(ClaimType claimTy, ModuleOp module) {
  assert(claimTy.isProven() && "getSubproofs() only valid on proven ClaimTypes");

  auto op = SymbolTable::lookupNearestSymbolFrom(module, claimTy.getProof());
  if (!op)
    llvm::report_fatal_error("getSubproofs: couldn't find referenced proof");

  // a leaf ImplOp has no subproofs
  if (isa<ImplOp>(op))
    return {};

  ProofOp proofOp = dyn_cast<ProofOp>(op);
  if (!proofOp)
    llvm::report_fatal_error("getSubproofs: expected proof to refer to a ProofOp");
  return proofOp.getSubproofClaims();
}

static LogicalResult recordProvenClaimAndSubproofs(
    ClaimType unprovenClaim,
    ClaimType provenClaim,
    ModuleOp module,
    DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // unprovenClaim must be unproven
  if (unprovenClaim.isProven()) {
    if (err) err() << "expected unproven claim";
    return failure();
  }

  // provenClaim must be proven
  if (!provenClaim.isProven()) {
    if (err) err() << "expected proven claim";
    return failure();
  }
  
  // early exit if we've already recorded this claim
  if (subst.count(unprovenClaim))
    return success();

  subst[unprovenClaim] = provenClaim;

  // get prerequisite claims from both sides
  auto requirements = getRequirements(unprovenClaim, module);
  auto subproofs = getSubproofs(provenClaim, module);

  if (requirements.size() != subproofs.size()) {
    if (err) err() << "arity mismatch: proof for " << unprovenClaim
                   << " has wrong number of subproofs";
    return failure();
  }

  // recursively record substitutions for subproofs
  for (auto [req, subproof] : llvm::zip(requirements, subproofs)) {
    if (failed(recordProvenClaimAndSubproofs(req, subproof, module, subst, err)))
      return failure();
  }

  return success();
}

LogicalResult ClaimType::unifyWith(
    Type ty,
    ModuleOp module,
    DenseMap<Type,Type>& subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // the driver normalizes both sides before calling us
  ClaimType other = mlir::dyn_cast<ClaimType>(ty);
  if (!other) {
    if (err) {
      err() << "expected !trait.claim, but found " << ty;
    }
    return failure();
  }

  auto selfApp = getTraitApplication();
  auto otherApp = other.getTraitApplication();

  // same trait?
  if (selfApp.getTrait() != otherApp.getTrait()) {
    if (err) err() << "trait mismatch: expected " << selfApp.getTrait()
                   << ", but found " << otherApp.getTrait();
    return failure();
  }

  // same arity?
  auto selfArgs = getTypeArgs();
  auto otherArgs = other.getTypeArgs();
  if (selfArgs.size() != otherArgs.size()) {
    if (err) err() << "arity mismatch: expected " << selfArgs.size()
                   << " type arguments, but found " << otherArgs.size();
    return failure();
  }

  // check proofs
  auto selfProof = getProof();
  auto otherProof = other.getProof();
  if (selfProof && otherProof && selfProof != otherProof) {
    if (err) err() << "proof mismatch: expected " << selfProof
                   << ", but found " << otherProof;
    return failure();
  }
  if (selfProof && !otherProof) {
    if (err) err() << "cannot substitute proven claim with unproven claim";
    return failure();
  }

  // recurse on each argument pair
  for (auto [selfArg, otherArg] : llvm::zip(selfArgs, otherArgs)) {
    if (failed(unifyTypes(selfArg, otherArg, module, subst, err)))
      return failure();
  }

  // recursively record whole-claim substitutions when:
  // 1. self is unproven, and
  // 2. other is proven
  // this ensures that all proven claims resulting from substitution application are always associated with proofs
  if (!isProven() && other.isProven()) {
    if (failed(recordProvenClaimAndSubproofs(*this, other, module, subst, err)))
      return failure();
  }

  return success();
}


/// Collect exactly the immediate child Types of `ty`. If `ty` has no sub‐elements,
/// returns an empty vector.
static SmallVector<Type, 4> getImmediateSubTypes(Type ty) {
  SmallVector<Type, 4> children;
  ty.walkImmediateSubElements(
      /*walkAttrsFn=*/[](Attribute) {
        // We don't need to collect Attributes here, so do nothing.
      },
      /*walkTypesFn=*/[&](Type subTy) {
        children.push_back(subTy);
      });
  return children;
}


LogicalResult unifyTypes(Type expected,
                         Type found,
                         ModuleOp moduleOp,
                         llvm::DenseMap<Type, Type> &subst,
                         llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize both types by applying the current substitution
  expected = applySubstitution(subst, expected);
  found = applySubstitution(subst, found);

  // if the normalized types are equal, unification succeeds
  if (expected == found)
    return success();

  // if either side can own its unification, let it
  if (auto lhs = dyn_cast<MonomorphizableTypeInterface>(expected))
    return lhs.unifyWith(found, moduleOp, subst, err);

  if (auto rhs = dyn_cast<MonomorphizableTypeInterface>(found))
    return rhs.unifyWith(expected, moduleOp, subst, err);

  // structural fallback: same constructor & arity, then recurse on children
  SmallVector<Type, 4> expectedKids = getImmediateSubTypes(expected);
  SmallVector<Type, 4> foundKids    = getImmediateSubTypes(found);

  // the constructor and arity of subelements both types must match before recursing
  if (expectedKids.size() != foundKids.size() ||
      expected.getTypeID() != found.getTypeID()) {
    if (err)
      err() << "type mismatch: expected " << expected
            << " but found " << found;
    return failure();
  }

  // Recurse on each child pair
  for (auto [l, r] : llvm::zip(expectedKids, foundKids)) {
    if (failed(unifyTypes(l, r, moduleOp, subst, err)))
      return failure();
  }

  return success();
}


LogicalResult unifyTypes(
    Type expectedTy,
    Type foundTy,
    ModuleOp moduleOp,
    llvm::DenseMap<Type,Type> &subst) {
  auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
  return unifyTypes(expectedTy, foundTy, moduleOp, subst, errFn);
}


LogicalResult unifyTypes(
    Type expectedTy,
    Type foundTy,
    ModuleOp moduleOp) {
  DenseMap<Type,Type> discardedSubst;
  return unifyTypes(expectedTy, foundTy, moduleOp, discardedSubst);
}


} // end mlir::trait
