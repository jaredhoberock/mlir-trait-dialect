#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the trait dialect with a context.
void traitRegisterDialect(MlirContext ctx);

/// Create a monomorphize-trait pass
MlirPass traitCreateMonomorphizePass();

/// Create a trait.trait operation
MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* typeParams, intptr_t numTypeParams);

/// Create a trait.impl operation
MlirOperation traitImplOpCreate(MlirLocation loc, MlirStringRef traitName,
                                MlirType* typeArgs, intptr_t numTypeArgs);

/// Create a trait.method.call operation
MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName,
                                      MlirStringRef methodName,
                                      MlirType methodFunctionType,
                                      MlirValue witness,
                                      MlirValue* arguments, intptr_t numArguments,
                                      MlirType* resultTypes, intptr_t numResults);

/// Create a trait.func.call operation
MlirOperation traitFuncCallOpCreate(MlirLocation loc,
                                    MlirStringRef callee,
                                    MlirType calleeFunctionType,
                                    MlirValue* arguments, intptr_t numArguments,
                                    MlirType* resultTypes, intptr_t numResults);

/// Create a trait.witness operation
MlirOperation traitWitnessOpCreate(MlirLocation loc, MlirStringRef traitName,
                                   MlirType* typeArgs, intptr_t numTypeArgs);

/// Return the !trait.poly<uniqueId> type
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int uniqueId);

// Return the !trait.witness<@Trait[Type1, Type2, ...]> type
MlirType traitWitnessTypeGet(MlirContext ctx,
                             MlirStringRef traitName,
                             MlirType* typeArgs, intptr_t numTypeArgs);

#ifdef __cplusplus
}
#endif
