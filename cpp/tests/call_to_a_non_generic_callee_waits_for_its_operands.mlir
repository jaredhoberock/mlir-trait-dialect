// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 >/dev/null | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Two non-generic callees whose parameter spells @H[i32]::E, one declared
// before @main and one after, each called with an operand whose producer
// spells the same projection. The stage settles each callee's signature and
// each operand's producer where it stands, in whichever order it reaches
// them, and a call compares the two spellings only once both are settled:
// both calls lower, nothing is named, and the program returns 7 + 7.

// QUIET-NOT: error
// CHECK: 14

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.impl private @H_i32(%s: !trait.claim<@H[i32]>) { trait.assoc_type @E = i64 }
func.func private @k_before(%x: !trait.proj<@H[i32], "E">) -> i64 {
  %y = builtin.unrealized_conversion_cast %x : !trait.proj<@H[i32], "E"> to i64
  return %y : i64
}
func.func @main() -> i64 {
  %c = arith.constant 7 : i64
  %p = builtin.unrealized_conversion_cast %c : i64 to !trait.proj<@H[i32], "E">
  %a = trait.func.call @k_before(%p) : (!trait.proj<@H[i32], "E">) -> i64
  %b = trait.func.call @k_after(%p) : (!trait.proj<@H[i32], "E">) -> i64
  %r = arith.addi %a, %b : i64
  return %r : i64
}
func.func private @k_after(%x: !trait.proj<@H[i32], "E">) -> i64 {
  %y = builtin.unrealized_conversion_cast %x : !trait.proj<@H[i32], "E"> to i64
  return %y : i64
}
