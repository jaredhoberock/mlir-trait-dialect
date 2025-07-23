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
  Type ty,
  ModuleOp module,
  llvm::function_ref<InFlightDiagnostic()> emitError) {
  // XXX if ty is a PolyType, should we unify with a different PolyType?
  return success();
}


LogicalResult ProofType::verify(function_ref<InFlightDiagnostic()> emitError,
                                FlatSymbolRefAttr /*traits*/,
                                ArrayRef<Type> typeArgs) {
  auto fail = [&]() -> LogicalResult {
    return emitError ? emitError() << "nested !trait.proof types are not allowed"
                     : failure();
  };

  for (Type t : typeArgs) {
    bool nested = false;
    t.walk([&](Type sub) {
      if (mlir::isa<ProofType>(sub))
        nested = true;
    });
    if (nested)
      return fail();
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


/// Attempt to unify a SymbolicTypeUnificationInterface `symTy` against another type `otherTy`.
/// If `symTy` already has a mapping in `substitution`, verify it matches `otherTy`. Otherwise,
/// ensure `otherTy` can be unified with `symTy` and record the mapping.
static LogicalResult unifySymbolicType(SymbolicTypeUnificationInterface symTy,
                                       Type otherTy,
                                       ModuleOp moduleOp,
                                       llvm::DenseMap<Type, Type> &substitution,
                                       llvm::function_ref<InFlightDiagnostic()> emitError) {
  // If we've already substituted a concrete type for symTy, check consistency.
  if (auto it = substitution.find(symTy); it != substitution.end()) {
    if (it->second != otherTy) {
      if (emitError)
        return emitError() << "mismatched substitution for type "
                           << symTy << ": expected "
                           << it->second << ", but found " << otherTy;

      return failure();
    }
    return success();
  }

  // no substitution already exists, check that we can unify
  if (failed(symTy.unifyWith(otherTy, moduleOp, emitError))) {
    return failure();
  }

  // Record the new substitution and succeed.
  substitution[symTy] = otherTy;
  return success();
}


LogicalResult unifyTypes(Type expectedTy,
                         Type foundTy,
                         ModuleOp moduleOp,
                         llvm::DenseMap<Type, Type> &substitution,
                         llvm::function_ref<InFlightDiagnostic()> emitError) {
  // normalize both types by applying the current substitution
  expectedTy = applySubstitution(substitution, expectedTy);
  foundTy = applySubstitution(substitution, foundTy);

  // if the normalized types are equal, unification succeeds
  if (expectedTy == foundTy)
    return success();

  // expectedTy is a SymbolicTypeUnificationInterface
  if (auto symTy = dyn_cast<SymbolicTypeUnificationInterface>(expectedTy)) {
    return unifySymbolicType(symTy,
                             foundTy,
                             moduleOp,
                             substitution,
                             emitError);
  }

  // foundTy is a SymbolicTypeUnificationInterface
  if (auto symTy = dyn_cast<SymbolicTypeUnificationInterface>(foundTy)) {
    return unifySymbolicType(symTy,
                             expectedTy,
                             moduleOp,
                             substitution,
                             emitError);
  }

  // recurse into sub elements
  SmallVector<Type, 4> expectedKids = getImmediateSubTypes(expectedTy);
  SmallVector<Type, 4> foundKids    = getImmediateSubTypes(foundTy);

  // the number of subelements and TypeID of both types must match before recursing
  if (expectedKids.size() != foundKids.size() ||
      expectedTy.getTypeID() != foundTy.getTypeID()) {
    if (emitError)
      emitError() << "type mismatch: expected " << expectedTy
                  << " but found " << foundTy;
    return failure();
  }

  // Recurse on each child pair
  for (unsigned i = 0, e = expectedKids.size(); i < e; ++i) {
    if (failed(unifyTypes(expectedKids[i],
                          foundKids[i],
                          moduleOp,
                          substitution,
                          emitError)))
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
