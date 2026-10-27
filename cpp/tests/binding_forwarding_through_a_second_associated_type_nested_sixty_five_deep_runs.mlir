// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | timeout 120 mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I binds A<U> to B<U>, which it binds to U: a binding that forwards through a
// second associated type, two steps a projection. !P65 nests a projection of A
// sixty-five deep in its own argument. A projection written in another's
// argument adds no depth to the chain it stands in, so every chain is two
// steps deep, though the nest takes 130 steps in all: the hypothesis @need
// states, the allegation proving it, and the application @C[!P65] selection
// reads all settle to i64, and @C_i64's method runs.

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @Tr(%s: !trait.claim<@Tr[!T]>) {
  trait.assoc_type @A<[!trait.poly<1>]>
  trait.assoc_type @B<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@Tr[i32]>) {
  trait.assoc_type @A<[!trait.poly<0>]> = !trait.proj<@Tr[i32], "B", [!trait.poly<0>]>
  trait.assoc_type @B<[!trait.poly<0>]> = !trait.poly<0>
}
trait.trait private @C(%s: !trait.claim<@C[!T]>) { trait.method @value() -> i64 }
trait.impl private @C_i64(%s: !trait.claim<@C[i64]>) { trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }

!P0 = i64
// REPEAT 0 64: !P{k+1} = !trait.proj<@Tr[i32], "A", [!P{k}]>

func.func @need(%e: !trait.claim<!P65 = i64>) { return }
func.func @main() -> i64 {
  %e = trait.allege !P65 = i64
  func.call @need(%e) : (!trait.claim<!P65 = i64>) -> ()
  %c = trait.allege @C[!P65]
  %v = trait.method.call %c @value() : !trait.claim<@C[!P65]>, () -> i64
  return %v : i64
}
