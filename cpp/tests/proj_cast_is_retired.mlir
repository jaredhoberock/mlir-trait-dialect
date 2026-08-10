// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// trait.proj.cast has been retired: value coercions and claim respells now lower
// to trait.coerce citing type equalities. A hand-written proj.cast names a custom
// op the trait dialect no longer registers, so it is rejected at parse time
// rather than silently accepted.

// CHECK: custom op 'trait.proj.cast' is unknown
func.func @proj_cast_is_retired(%v: i64) -> i64 {
  %r = trait.proj.cast %v, %v : i64 to i64 by i64
  return %r : i64
}
