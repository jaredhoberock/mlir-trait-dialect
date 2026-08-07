// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Congruence closure unites; it never decomposes. An equality between two
// wrapped spellings, tuple<i64> = tuple<i32>, does not entail i64 = i32, so a
// coerce that would read it backwards through the tuple head is refused.

func.func @no_decomposition(%v: i64, %e: !trait.claim<tuple<i64> = tuple<i32>>) -> i32 {
  // expected-error @below {{are not equal under the cited equalities}}
  %c = trait.coerce %v : i64 to i32 via (%e) : (!trait.claim<tuple<i64> = tuple<i32>>)
  return %c : i32
}
