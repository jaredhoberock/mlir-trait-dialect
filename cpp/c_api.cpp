#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::trait;

extern "C" {

void traitRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<TraitDialect>();
}

MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);
  auto op = builder.create<TraitOp>(
    unwrap(loc),
    builder.getStringAttr(StringRef(name.data, name.length))
  );
  return wrap(op.getOperation());
}

MlirOperation traitImplOpCreate(MlirLocation loc, MlirStringRef traitName, MlirType concreteType) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);
  auto op = builder.create<ImplOp>(
    unwrap(loc), 
    FlatSymbolRefAttr::get(ctx, StringRef(traitName.data, traitName.length)),
    TypeAttr::get(unwrap(concreteType))
  );
  return wrap(op.getOperation());
}

MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName, MlirType selfType,
                                      MlirStringRef methodName,
                                      MlirValue* arguments, intptr_t numArguments,
                                      MlirType* resultTypes, intptr_t numResults) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> args;
  for (intptr_t i = 0; i < numArguments; ++i) {
    args.push_back(unwrap(arguments[i]));
  }

  SmallVector<Type> results;
  for (intptr_t i = 0; i < numResults; ++i) {
    results.push_back(unwrap(resultTypes[i]));
  }

  auto op = builder.create<MethodCallOp>(
    unwrap(loc),
    results,
    StringRef(traitName.data, traitName.length),
    StringRef(methodName.data, methodName.length),
    TypeAttr::get(unwrap(selfType)),
    args
  );

  return wrap(op.getOperation());
}

MlirOperation traitFuncCallOpCreate(MlirLocation loc,
                                    MlirStringRef callee,
                                    MlirType calleeFunctionType,
                                    MlirValue* arguments, intptr_t numArguments,
                                    MlirType* resultTypes, intptr_t numResults) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> args;
  for (intptr_t i = 0; i < numArguments; ++i) {
    args.push_back(unwrap(arguments[i]));
  }

  SmallVector<Type> results;
  for (intptr_t i = 0; i < numResults; ++i) {
    results.push_back(unwrap(resultTypes[i]));
  }

  auto op = builder.create<FuncCallOp>(
    unwrap(loc),
    results,
    FlatSymbolRefAttr::get(ctx, StringRef(callee.data, callee.length)),
    TypeAttr::get(unwrap(calleeFunctionType)),
    args
  );

  return wrap(op.getOperation());
}

MlirType traitSelfTypeGet(MlirContext ctx) {
  return wrap(SelfType::get(unwrap(ctx)));
}

MlirType traitPolyTypeGet(MlirContext wrappedCtx,
                          unsigned int uniqueId,
                          MlirStringRef* traitBounds,
                          intptr_t numTraitBounds) {
  MLIRContext* ctx = unwrap(wrappedCtx);

  // build an array of FlatSymbolRefAttr from the C strings
  SmallVector<FlatSymbolRefAttr> bounds;
  bounds.reserve(numTraitBounds);
  for (intptr_t i = 0; i < numTraitBounds; ++i) {
    auto &sref = traitBounds[i];
    StringRef name(sref.data, sref.length);
    bounds.push_back(FlatSymbolRefAttr::get(ctx, name));
  }

  return wrap(PolyType::get(ctx, uniqueId, bounds));
}

} // end extern "C"
