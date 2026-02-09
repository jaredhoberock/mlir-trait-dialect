// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_OP_INTERFACES_TD
#define TRAIT_OP_INTERFACES_TD

include "mlir/IR/OpBase.td"

/// Operation interface for ops whose result types can be specialized
/// once operands become monomorphic.
///
/// Contract:
///   - Return failure if result types cannot be specialized yet.
///   - Return a SmallVector<Type> of refined result types if specialization
///     is possible.
///   - The number of result types must match the op's existing arity.
///
/// The rewrite pattern in the trait dialect will use this to update an
/// operation's result types as its operands become monomorphic.
def ResultTypeSpecializationOpInterface : OpInterface<"ResultTypeSpecializationOpInterface"> {
  let cppNamespace = "::mlir::trait";

  let description = [{
    Interface for operations whose result types can be specialized once
    their operands become monomorphic.
  
    This allows a generic rewrite to refine polymorphic result types
    incrementally as specialization information becomes available,
    without requiring each op to implement its own rewrite pattern.
  }];

  let methods = [
    InterfaceMethod<
      /*desc*/
      "Attempt to refine/specialize the result types based on the "
      "current operand types. Return failure() if no refinement is "
      "possible, otherwise return a SmallVector<Type> of the new "
      "result types. This must not mutate the operation.",
      /*retType*/
      "mlir::FailureOr<llvm::SmallVector<mlir::Type>>",
      /*methodName*/
      "specializeResultTypes"
    >
  ];
}

#endif // TRAIT_OP_INTERFACES_TD
