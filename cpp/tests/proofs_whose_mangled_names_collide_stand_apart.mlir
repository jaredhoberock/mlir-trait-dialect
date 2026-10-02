// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A mangled proof name is not one-to-one: @I at i32 and @I_hf83e4f0dfbbf75af,
// an impl of @B with no parameters, mangle to one proof name, which a function
// of the module also holds. Proving @A[i32] reserves a free name while its
// premise @B[i32] is proven, the premise's proof takes another, and no proof
// takes the function's: the program verifies and runs, adding the 9 @f
// returns to the 3 the function does.

// CHECK: {{^}}12{{$}}

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {}
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.impl private @I_hf83e4f0dfbbf75af(%self: !trait.claim<@B[i32]>, %eq: !trait.claim<i32 = i32>) {}
trait.impl private @I(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>) {}
func.func private @f(%a: !trait.claim<@A[!T]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
func.func private @I_hf83e4f0dfbbf75af_p() -> i64 {
  %n = arith.constant 3 : i64
  return %n : i64
}
func.func @main() -> i64 {
  %a = trait.allege @A[i32]
  %v = trait.func.call @f(%a) : (!trait.claim<@A[i32]>) -> i64
  %w = func.call @I_hf83e4f0dfbbf75af_p() : () -> i64
  %s = arith.addi %v, %w : i64
  return %s : i64
}
