// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A declaration's block is a dominance region: a value its block computes is
// read only after its definition.

!T = !trait.poly<0>
trait.trait private @C(%self: !trait.claim<@C[!T]>) {}
trait.trait private @B(%self: !trait.claim<@B[!T]>) -> !trait.claim<@C[!T]> {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@C[!T]> {}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) {}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  %c = trait.witness @C_i32 for @C[i32]
  trait.return %c : !trait.claim<@C[i32] by @C_i32>
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  // expected-error @+1 {{operand #0 does not dominate this use}}
  %c = trait.project %b[0] : !trait.claim<@B[i32] by @B_i32> -> !trait.claim<@C[i32] by @C_i32>
  // expected-note @+1 {{operand defined here}}
  %b = trait.witness @B_i32 for @B[i32]
  trait.return %c : !trait.claim<@C[i32] by @C_i32>
}
