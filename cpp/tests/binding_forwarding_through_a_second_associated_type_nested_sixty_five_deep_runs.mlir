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
trait.trait private @Tr(!trait.claim<@Tr[!T]>) assoc {A = 1, B = 1}
trait.impl private @I(%s: !trait.claim<@Tr[i32]>) bindings {A = !trait.proj<@Tr[i32], "B", [!trait.poly<0>]>, B = !trait.poly<0>} {
}
trait.trait private @C(!trait.claim<@C[!T]>) members [@C.value]
func.func private @C.value(!trait.claim<@C[!T]>) -> i64
trait.impl private @C_i64(%s: !trait.claim<@C[i64]>) members [@C.value -> @C_i64.value] {
}
func.func private @C_i64.value(%s: !trait.claim<@C[i64]>) -> i64 {
  %c = arith.constant 7 : i64 return %c : i64
}

!P0 = i64
// REPEAT 0 64: !P{k+1} = !trait.proj<@Tr[i32], "A", [!P{k}]>

func.func @need(%e: !trait.claim<!P65 = i64>) { return }
func.func @main() -> i64 {
  %e = trait.allege !P65 = i64
  func.call @need(%e) : (!trait.claim<!P65 = i64>) -> ()
  %c = trait.allege @C[!P65]
  %v = trait.func.call @C.value(%c) : (!trait.claim<@C[!P65]>) -> i64
  return %v : i64
}
