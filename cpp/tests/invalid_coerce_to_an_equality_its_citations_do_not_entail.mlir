// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A coerce to an equality claim is admissible only when the ground congruence
// closure of the equalities it cites entails the result. The cited i32 = i64
// does not relate i32 to the unrelated f32, so the closure over the endpoints
// and the single leaf keeps i32 and f32 in different classes and the coerce of
// the refl witness is refused: it may not name an equality its citations do not
// entail.

func.func @unentailed(%p: !trait.claim<i32 = i64>) -> !trait.claim<i32 = f32> {
  %r = trait.witness refl : !trait.claim<i32 = i32>
  // expected-error @below {{input type '!trait.claim<i32 = i32>' and result type '!trait.claim<i32 = f32>' are not equal under the cited equalities}}
  %c = trait.coerce %r : !trait.claim<i32 = i32> to !trait.claim<i32 = f32> via (%p) : (!trait.claim<i32 = i64>)
  return %c : !trait.claim<i32 = f32>
}
