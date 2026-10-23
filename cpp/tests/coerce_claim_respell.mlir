// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A claim-respell coerce: the operand is an application claim, and the result is
// the SAME application with one argument respelled through a cited child
// equality. This is the shape a bridge cast produces when it respells a claim's
// trait application. The result names no proof: the operand's proves the
// operand's spelling, and a reader of evidence finds it through the coercion.
// The congruence closure lifts the child equality through the application
// argument.

!S = !trait.poly<0>

trait.trait private @Conv(%self: !trait.claim<@Conv[!S]>) {
  trait.assoc_type @At
}

trait.trait private @Safe(%self: !trait.claim<@Safe[!trait.poly<0>, !trait.poly<1>]>) {
}

trait.impl private @Conv_i1(%self: !trait.claim<@Conv[i1]>) {
  trait.assoc_type @At = i64
}

trait.impl private @Safe_impl(%self: !trait.claim<@Safe[i32, i64]>) {
}

trait.proof private @Safe_proof {
  %d = trait.derive @Safe[i32, i64] from @Safe_impl given()
  trait.return %d : !trait.claim<@Safe[i32, i64]>
}

// CHECK-LABEL: func.func @respell
// CHECK: trait.coerce %{{.*}} : !trait.claim<@Safe[i32, i64] by @Safe_proof> to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
func.func @respell()
    -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]> {
  %safe = trait.witness @Safe_proof for @Safe[i32, i64]
  %eq = trait.witness proj_resolve !trait.proj<@Conv[i1], "At"> resolves i64 by @Conv_i1
    : !trait.claim<!trait.proj<@Conv[i1], "At"> = i64>
  %c = trait.coerce %safe : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
    via (%eq) : (!trait.claim<!trait.proj<@Conv[i1], "At"> = i64>)
  return %c : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
}
