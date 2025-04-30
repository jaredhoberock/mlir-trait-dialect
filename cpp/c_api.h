#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the trait dialect with a context.
void traitRegisterDialect(MlirContext ctx);

/// Create a trait.trait operation
MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name);

/// Create a trait.impl operation
MlirOperation traitImplOpCreate(MlirLocation loc, MlirStringRef traitName, MlirType concreteType);

/// Create a trait.method.call operation
MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName, MlirType selfType,
                                      MlirStringRef methodName,
                                      MlirValue* arguments, intptr_t numArguments,
                                      MlirType* resultTypes, intptr_t numResults);

/// Create a trait.func.call operation
MlirOperation traitFuncCallOpCreate(MlirLocation loc,
                                    MlirStringRef callee,
                                    MlirValue* arguments, intptr_t numArguments,
                                    MlirType* resultTypes, intptr_t numResults);

/// Return the !trait.self type
MlirType traitSelfTypeGet(MlirContext ctx);

/// Return the !trait.poly<uniqueId, [@Trait1, @Trait2, ...]> type
MlirType traitPolyTypeGet(MlirContext ctx,
                          unsigned int uniqueId,
                          MlirStringRef* traitBounds, intptr_t numTraitBounds);

#ifdef __cplusplus
}
#endif
