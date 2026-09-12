// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A witness carries the claim the evidence it names stands over. @p proves
// @B[i32]; a witness for @B[i64] citing it carries a claim no evidence stands
// over. Reading the cited impl's header alone accepts it, because @B_blanket
// covers every application of @B, so the proof's own claim is what the witness
// is read against.

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {}
trait.impl private @A_i32 for @A[i32] {}
trait.impl private @B_blanket for @B[!trait.poly<0>] {}
trait.proof private @p proves @B_blanket for @B[i32] given [@A_i32]

func.func @main() -> !trait.claim<@B[i64] by @p> {
  // expected-error @below {{the proof @p this witness cites stands over another claim: type mismatch: expected '!trait.claim<@B[i32]>' but found '!trait.claim<@B[i64]>'}}
  %w = trait.witness @p for @B[i64]
  return %w : !trait.claim<@B[i64] by @p>
}
