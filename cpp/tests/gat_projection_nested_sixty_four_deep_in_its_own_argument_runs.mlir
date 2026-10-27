// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | timeout 120 mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I binds Item<U> to U, and !P64 is a projection of Item nested sixty-four
// deep in its own argument. A projection's argument is read before the
// projection is, and the binding is specialized at what was read, so the
// binding holds no argument still to read and each level is read once: the
// hypothesis @need states, the allegation proving it, and the application
// @B[!P64] selection reads all settle to i64, and @B_i64's method runs. A
// reading that read each level's argument again inside its binding would make
// 2^64 readings; the compile is bounded so that such a reading fails the row
// rather than holding the suite.

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.poly<0>
}
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.method @value() -> i64 }
trait.impl private @B_i64(%s: !trait.claim<@B[i64]>) { trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }

!P0 = i64
// REPEAT 0 63: !P{k+1} = !trait.proj<@A[i32], "Item", [!P{k}]>

func.func @need(%e: !trait.claim<!P64 = i64>) { return }
func.func @main() -> i64 {
  %e = trait.allege !P64 = i64
  func.call @need(%e) : (!trait.claim<!P64 = i64>) -> ()
  %b = trait.allege @B[!P64]
  %v = trait.method.call %b @value() : !trait.claim<@B[!P64]>, () -> i64
  return %v : i64
}
