// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A witness names @Box_i64, which proves @Box[i64], for a claim spelling
// @Box[@Gen[i64]::A]. @Gen has no impl, so that projection is equal to itself
// alone: nothing carries the impl's header to the claim, and the witness is
// refused where it stands. The sibling row is the same witness once @Gen binds
// the projection.

trait.trait private @Gen[!trait.poly<0>] {
  trait.assoc_type @A
}

trait.trait private @Box[!trait.poly<1>] {}

trait.impl private @Box_i64 for @Box[i64] {}

func.func @main() {
  // expected-error @below {{type mismatch: expected '!trait.claim<@Box[i64]>' but found '!trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>'}}
  %w = trait.witness @Box_i64 for @Box[!trait.proj<@Gen[i64], "A">]
  return
}
