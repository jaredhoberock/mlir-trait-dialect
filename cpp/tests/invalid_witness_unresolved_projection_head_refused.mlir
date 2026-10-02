// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A witness names @Box_i64, which proves @Box[i64], for a claim spelling
// @Box[@Gen[i64]::A]. @Gen has no impl, so that projection is equal to itself
// alone: nothing carries the impl's header to the claim. Verification leaves a
// spelling only selection's settlement could decide to the stage, and the
// stage refuses the witness the call dispatches through. The sibling row is
// the same witness once @Gen binds the projection.

trait.trait private @Gen(%self: !trait.claim<@Gen[!trait.poly<0>]>) {
  trait.assoc_type @A
}

trait.trait private @Box(%self: !trait.claim<@Box[!trait.poly<1>]>) {
  trait.method @v() -> i64
}

trait.impl private @Box_i64(%self: !trait.claim<@Box[i64]>) {
  trait.method @v() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}

func.func @main() -> i64 {
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %w = trait.witness @Box_i64 for @Box[!trait.proj<@Gen[i64], "A">]
  %r = trait.method.call %w @Box[!trait.proj<@Gen[i64], "A">]::@v() : () -> i64 by @Box_i64
  return %r : i64
}
