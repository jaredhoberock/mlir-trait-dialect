// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// An equality witness is a leaf: it cites the impl that resolves a projection,
// or it carries the refl marker. One with neither states an equality nothing it
// carries establishes, even where the equality it is given entails it; that
// equality reaches the result through a coerce of a refl witness citing it.

func.func @neither(%p: !trait.claim<i32 = i64>) -> !trait.claim<i64 = i32> {
  // expected-error @below {{an equality witness cites an impl resolving a projection or carries a refl marker}}
  %c = "trait.witness"(%p) : (!trait.claim<i32 = i64>) -> !trait.claim<i64 = i32>
  return %c : !trait.claim<i64 = i32>
}
