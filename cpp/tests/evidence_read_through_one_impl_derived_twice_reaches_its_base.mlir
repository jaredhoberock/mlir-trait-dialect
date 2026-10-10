// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @P_gen at @P[i64] is derived twice, over different evidence: %p1 given
// @E_mid's derivation, whose return projects %p2, and %p2 given @E_base's,
// whose return is a derive. The reading meets @P_gen's requirement at @P[i64]
// twice, but through two derivations, so it is finite: @P_gen, @E_mid, @P_gen,
// @E_base, then a base, and @B_i64's method runs.

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) {
  trait.method @v() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 }
}
trait.trait private @E(%self: !trait.claim<@E[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @P(%self: !trait.claim<@P[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @E_base(%self: !trait.claim<@E[i64]>) {
  %b = trait.derive @B[i64] from @B_i64 given()
  trait.return %b : !trait.claim<@B[i64]>
}
trait.impl private @P_gen(%self: !trait.claim<@P[!T]>, %e: !trait.claim<@E[!T]>) {
  %b = trait.project %e[0] : !trait.claim<@E[!T]> -> !trait.claim<@B[!T]>
  trait.return %b : !trait.claim<@B[!T]>
}
trait.impl private @E_mid(%self: !trait.claim<@E[i64]>, %p: !trait.claim<@P[i64]>) {
  %b = trait.project %p[0] : !trait.claim<@P[i64]> -> !trait.claim<@B[i64]>
  trait.return %b : !trait.claim<@B[i64]>
}
func.func @main() -> i64 {
  %eb = trait.derive @E[i64] from @E_base given()
  %p2 = trait.derive @P[i64] from @P_gen[i64] given(%eb) : (!trait.claim<@E[i64]>)
  %em = trait.derive @E[i64] from @E_mid given(%p2) : (!trait.claim<@P[i64]>)
  %p1 = trait.derive @P[i64] from @P_gen[i64] given(%em) : (!trait.claim<@E[i64]>)
  %b = trait.project %p1[0] : !trait.claim<@P[i64]> -> !trait.claim<@B[i64]>
  %v = trait.method.call %b @B[i64]::@v() : () -> i64
  return %v : i64
}
