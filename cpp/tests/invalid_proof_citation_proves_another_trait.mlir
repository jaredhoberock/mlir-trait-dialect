// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The requirement @B[i32] states is @A[i32]. Evidence of an impl of an
// unrelated trait stands for nothing there, however well that impl's header
// carries to its own claim.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @C(%self: !trait.claim<@C[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {}
trait.impl private @C_f32(%self: !trait.claim<@C[f32]>) {}

// expected-error @below {{returns '!trait.claim<@C[f32] by @C_f32>' for requirement 0, which trait '@B' states as '!trait.claim<@A[i32]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  %c = trait.witness @C_f32 for @C[f32]
  trait.return %c : !trait.claim<@C[f32] by @C_f32>
}
