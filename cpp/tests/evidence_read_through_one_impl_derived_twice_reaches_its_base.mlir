// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @P_gen at @P[i64] is derived twice, over different evidence: %p1 given
// @E_mid's derivation, whose return projects %p2, and %p2 given @E_base's,
// whose return is a derive. The reading meets @P_gen's requirement at @P[i64]
// twice, but through two derivations, so it is finite: @P_gen, @E_mid, @P_gen,
// @E_base, then a base, and @B_i64's member runs.

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!T]>) members [@B.v]
func.func private @B.v(!trait.claim<@B[!T]>) -> i64
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) members [@B.v -> @B_i64.v] {
}
func.func private @B_i64.v(%self: !trait.claim<@B[i64]>) -> i64 {
  %c = arith.constant 7 : i64 return %c : i64
}
trait.trait private @E(!trait.claim<@E[!T]>) -> !trait.claim<@B[!T]>
trait.trait private @P(!trait.claim<@P[!T]>) -> !trait.claim<@B[!T]>
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
  %v = trait.func.call @B.v(%b) : (!trait.claim<@B[i64]>) -> i64
  return %v : i64
}
