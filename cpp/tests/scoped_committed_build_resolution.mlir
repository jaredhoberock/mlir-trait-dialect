// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A committed-fact substitution build (here the witness verifier's self-claim
// match) resolves a ground projection its own substitution mints, then compares the
// resolved value. @Gen[i64]::A binds to i32 through @Gen's unique impl, so:
//   - a witness for @T[@Gen[i64]::A] backed by a proof of @T[i32] resolves the
//     projection to i32 and matches (accepted);
//   - a witness for @T[@Gen[i64]::A] backed by a proof of @T[f32] resolves the
//     projection to i32 and finds it distinct from f32 (rejected).
// The resolution is module-capable and premise-blind: it reads @Gen's one impl
// binding, nothing more.

trait.trait @Gen[!trait.poly<0>] {
  trait.assoc_type @A
}

trait.impl @Gen_i64 for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.trait @T[!trait.poly<1>] {}

trait.impl @T_i32 for @T[i32] {}
trait.impl @T_f32 for @T[f32] {}

func.func @resolves_and_matches() {
  %w = trait.witness @T_i32 for @T[!trait.proj<@Gen[i64], "A">]
  return
}

func.func @resolves_and_rejects() {
  // expected-error @below {{type mismatch: expected 'i32' but found 'f32'}}
  %w = trait.witness @T_f32 for @T[!trait.proj<@Gen[i64], "A">]
  return
}
