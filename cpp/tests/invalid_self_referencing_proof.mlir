// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proof witnessing itself stands for its own claim, which supplies a premise
// only when the premise is that claim. Here the premise is @A[i32] and the
// proof proves @B[i32].

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {}

trait.impl private @B_impl(%self: !trait.claim<@B[i32]>, %a: !trait.claim<@A[i32]>) {
  trait.return %a : !trait.claim<@A[i32]>
}

trait.impl private @A_impl(%self: !trait.claim<@A[i32]>) {}

trait.proof private @self_proof {
  %w = trait.witness @self_proof for @B[i32]
  // expected-error @+1 {{premise 0 of impl '@B_impl' is '!trait.claim<@A[i32]>', and the derive supplies '!trait.claim<@B[i32]>'}}
  %d = trait.derive @B[i32] from @B_impl given(%w) : (!trait.claim<@B[i32] by @self_proof>)
  trait.return %d : !trait.claim<@B[i32]>
}
