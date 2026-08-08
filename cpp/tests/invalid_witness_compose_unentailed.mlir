// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A composition witness is admissible only when its premises' ground congruence
// closure entails the result equality. The premise i32 = i64 does not relate i32
// to the unrelated f32, so the closure over the endpoints and the single leaf
// keeps i32 and f32 in different classes and the witness is refused. This is the
// same closure trait.coerce replays; a composition may not name an equality its
// leaves do not entail.

func.func @unentailed(%p: !trait.claim<i32 = i64>) -> !trait.claim<i32 = f32> {
  // expected-error @+1 {{the premises do not entail 'i32' = 'f32'}}
  %c = trait.witness compose(%p) : (!trait.claim<i32 = i64>) : !trait.claim<i32 = f32>
  return %c : !trait.claim<i32 = f32>
}
