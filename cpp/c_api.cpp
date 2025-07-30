#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include "Passes.hpp"
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

MlirPass traitCreateMonomorphizePass() {
  return wrap(createMonomorphizePass().release());
}

MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* typeParams, intptr_t numTypeParams) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  SmallVector<Attribute> typeAttrs;
  for (intptr_t i = 0; i < numTypeParams; ++i) {
    typeAttrs.push_back(TypeAttr::get(unwrap(typeParams[i])));
  }

  OpBuilder builder(ctx);
  auto op = builder.create<TraitOp>(
    unwrap(loc),
    builder.getStringAttr(StringRef(name.data, name.length)),
    builder.getArrayAttr(typeAttrs)
  );

  return wrap(op.getOperation());
}

MlirOperation traitImplOpCreate(MlirLocation loc, MlirStringRef traitName,
                                MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  SmallVector<Attribute> typeAttrs;
  for (intptr_t i = 0; i < numTypeArgs; ++i) {
    typeAttrs.push_back(TypeAttr::get(unwrap(typeArgs[i])));
  }

  OpBuilder builder(ctx);
  auto op = builder.create<ImplOp>(
    unwrap(loc),
    FlatSymbolRefAttr::get(ctx, StringRef(traitName.data, traitName.length)),
    builder.getArrayAttr(typeAttrs)
  );

  return wrap(op.getOperation());
}

MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName,
                                      MlirStringRef methodName,
                                      MlirType methodFunctionType,
                                      MlirValue proof,
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
    TypeAttr::get(unwrap(methodFunctionType)),
    unwrap(proof),
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

MlirOperation traitWitnessOpCreate(MlirLocation loc,
                                   MlirStringRef traitName,
                                   MlirType* typeArgs, intptr_t numTypeArgs,
                                   MlirValue* wrappedPrereqs, intptr_t numPrereqs) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  MlirType wrappedProofType = traitProofTypeGet(wrap(ctx), traitName, typeArgs, numTypeArgs);

  OpBuilder builder(ctx);

  SmallVector<Value> prereqs;
  for (intptr_t i = 0; i < numPrereqs; ++i) {
    prereqs.push_back(unwrap(wrappedPrereqs[i]));
  }

  auto op = builder.create<WitnessOp>(
    unwrap(loc),
    unwrap(wrappedProofType),
    prereqs
  );

  return wrap(op.getOperation());
}

MlirOperation traitProjectOpCreate(MlirLocation loc,
                                   MlirValue srcProof,
                                   MlirStringRef traitName,
                                   MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  MlirType wrappedProofType = traitProofTypeGet(wrap(ctx), traitName, typeArgs, numTypeArgs);

  OpBuilder builder(ctx);

  auto op = builder.create<ProjectOp>(
    unwrap(loc),
    unwrap(wrappedProofType),
    unwrap(srcProof)
  );

  return wrap(op.getOperation());
}

MlirOperation traitAssumeOpCreate(MlirLocation loc,
                                  MlirStringRef traitName,
                                  MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  MlirType wrappedProofType = traitProofTypeGet(wrap(ctx), traitName, typeArgs, numTypeArgs);

  OpBuilder builder(ctx);

  auto op = builder.create<AssumeOp>(
    unwrap(loc),
    unwrap(wrappedProofType)
  );

  return wrap(op.getOperation());
}

MlirType traitPolyTypeGet(MlirContext wrappedCtx,
                          unsigned int uniqueId) {
  return wrap(PolyType::get(unwrap(wrappedCtx), uniqueId));
}

MlirType traitProofTypeGet(MlirContext wrappedCtx,
                           MlirStringRef traitName,
                           MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext* ctx = unwrap(wrappedCtx);
  FlatSymbolRefAttr traitRefAttr = FlatSymbolRefAttr::get(ctx, StringRef(traitName.data, traitName.length));

  SmallVector<Type> typeArgsVec;
  typeArgsVec.reserve(numTypeArgs);
  for (intptr_t i = 0; i < numTypeArgs; ++i) {
    typeArgsVec.push_back(unwrap(typeArgs[i]));
  }

  return wrap(ProofType::get(ctx, traitRefAttr, typeArgsVec));
}

} // end extern "C"
