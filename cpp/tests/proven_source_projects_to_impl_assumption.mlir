// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A proven claim's requirements continue past its trait's into the assumptions
// of the impl its proof cites, so @T_impl's own @U assumption stands at index 0
// of a claim of @T[i64] by @T_p (the trait requires nothing). The result carries
// the subproof that discharged it, read out of the proof by position.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @U(%self: !trait.claim<@U[!trait.poly<0>]>) {}
trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @U_impl(%self: !trait.claim<@U[i64]>) {}
trait.impl private @T_impl(%self: !trait.claim<@T[!trait.poly<0>]>, %u: !trait.claim<@U[!trait.poly<0>]>) {}
trait.proof private @T_p {
  %p0 = trait.witness @U_impl for @U[i64]
  %d = trait.derive @T[i64] from @T_impl[i64] given(%p0) : (!trait.claim<@U[i64] by @U_impl>)
  trait.return %d : !trait.claim<@T[i64]>
}

// CHECK: trait.project %{{.*}}[0] : <@T[i64] by @T_p> -> <@U[i64] by @U_impl>
func.func @f(%s: !trait.claim<@T[i64] by @T_p>) -> !trait.claim<@U[i64] by @U_impl> {
  %u = trait.project %s[0] : !trait.claim<@T[i64] by @T_p> -> !trait.claim<@U[i64] by @U_impl>
  return %u : !trait.claim<@U[i64] by @U_impl>
}
