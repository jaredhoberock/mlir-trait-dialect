// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | timeout 120 mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I binds S0<U> to S1<U>, and so on to S7<U>, which is a one-tuple of U: a
// chain of eight steps. !P17 nests a projection of S0 seventeen deep in its own
// argument. A projection written in another's argument stands as deep as the
// one it is written in, as rustc counts a normalization's depth, so each of the
// seventeen chains stands eight steps deep, far inside the depth limit, though
// the nest takes 136 steps in all: the hypothesis @need states, the allegation
// proving it, and the application @B[!P17] selection reads all settle to a
// one-tuple seventeen deep, and @B_n17's method runs.

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
// REPEAT 0 7:   trait.assoc_type @S{k}<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
// REPEAT 0 6:   trait.assoc_type @S{k}<[!trait.poly<0>]> = !trait.proj<@A[i32], "S{k+1}", [!trait.poly<0>]>
  trait.assoc_type @S7<[!trait.poly<0>]> = tuple<!trait.poly<0>>
}

!P0 = i64
// REPEAT 0 16: !P{k+1} = !trait.proj<@A[i32], "S0", [!P{k}]>
!N0 = i64
// REPEAT 0 16: !N{k+1} = tuple<!N{k}>

trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.method @value() -> i64 }
trait.impl private @B_n17(%s: !trait.claim<@B[!N17]>) { trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }

func.func @need(%e: !trait.claim<!P17 = !N17>) { return }
func.func @main() -> i64 {
  %e = trait.allege !P17 = !N17
  func.call @need(%e) : (!trait.claim<!P17 = !N17>) -> ()
  %b = trait.allege @B[!P17]
  %v = trait.method.call %b @B[!P17]::@value() : () -> i64
  return %v : i64
}
