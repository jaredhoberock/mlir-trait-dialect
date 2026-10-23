// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A coerce cites equality claims. An application claim is not equality evidence
// and carries no endpoints for the closure to seed, so citing one is refused.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}

func.func @application_citation(%p: !trait.claim<@T[i32]>) -> !trait.claim<i32 = i32> {
  %r = trait.witness refl : !trait.claim<i32 = i32>
  // expected-error @below {{coerce cites equality claims, but operand has type '!trait.claim<@T[i32]>'}}
  %c = trait.coerce %r : !trait.claim<i32 = i32> to !trait.claim<i32 = i32> via (%p) : (!trait.claim<@T[i32]>)
  return %c : !trait.claim<i32 = i32>
}
