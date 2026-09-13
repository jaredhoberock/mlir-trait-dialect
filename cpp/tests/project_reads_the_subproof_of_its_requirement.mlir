// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A proven claim's application requirement carries the provider of the subproof
// discharging it, read out of the proof by position: @T_p names @A_i64 then
// @B_i64 for @T's two requirements, so the hop at 0 carries @A_i64 and the hop
// at 1 carries @B_i64. Nothing here matches a spelling -- swap the two
// annotations and each hop refuses.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<1>] {}
trait.trait private @T[!trait.poly<2>] where [@A[!trait.poly<2>], @B[!trait.poly<2>]] {}

trait.impl private @A_i64 for @A[i64] {}
trait.impl private @B_i64 for @B[i64] {}
trait.impl private @T_impl for @T[i64] {}
trait.proof private @T_p proves @T_impl for @T[i64] given [@A_i64, @B_i64]

// CHECK-LABEL: func.func @f
// CHECK: trait.project %arg0[0] : <@T[i64] by @T_p> -> <@A[i64] by @A_i64>
// CHECK: trait.project %arg0[1] : <@T[i64] by @T_p> -> <@B[i64] by @B_i64>
func.func @f(%s: !trait.claim<@T[i64] by @T_p>) -> !trait.claim<@B[i64] by @B_i64> {
  %a = trait.project %s[0]
    : !trait.claim<@T[i64] by @T_p> -> !trait.claim<@A[i64] by @A_i64>
  %b = trait.project %s[1]
    : !trait.claim<@T[i64] by @T_p> -> !trait.claim<@B[i64] by @B_i64>
  return %b : !trait.claim<@B[i64] by @B_i64>
}
