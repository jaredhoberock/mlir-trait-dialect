// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A derive derives its impl's header, which is a trait application, so a
// result claim holding an equality is refused.

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}

trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {}

func.func @derives_an_equality() {
  // expected-error @+1 {{result #0 must be a '!trait.claim' of a trait application, but got '!trait.claim<i32 = i32>'}}
  %c = "trait.derive"() <{impl = @B_i32, impl_args = []}> : () -> !trait.claim<i32 = i32>
  return
}
