// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A trait.project from a PROVEN source claim to one of the impl's own
// application assumptions verifies: the assumption is a candidate projection of
// the proven self, spelled proven by the subproof that discharged it. A proven
// source's candidates carry the proof each assumption was discharged by, so an
// impl assumption stands among them in its proven spelling and not only in its
// unproven one.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @U[!trait.poly<0>] {}
trait.trait private @T[!trait.poly<1>] {}
trait.impl private @U_impl for @U[i64] {}
trait.impl private @T_impl for @T[!trait.poly<2>] where [@U[!trait.poly<2>]] {}
trait.proof private @T_p proves @T_impl for @T[i64] given [@U_impl]

// CHECK: trait.project %{{.*}} to @U[i64] by @U_impl
func.func @f(%s: !trait.claim<@T[i64] by @T_p>) -> !trait.claim<@U[i64] by @U_impl> {
  %u = trait.project %s : @T[i64] by @T_p to @U[i64] by @U_impl
  return %u : !trait.claim<@U[i64] by @U_impl>
}
