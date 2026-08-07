// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -canonicalize | FileCheck %s

// The zero-evidence reflexive coerce is the discharged terminal state: input
// and result are identical, so it folds to its operand and disappears. Any
// cited evidence then dies by ordinary DCE.

// CHECK-LABEL: func.func @refl
// CHECK-NOT: trait.coerce
// CHECK: return %arg0 : i64
func.func @refl(%v: i64) -> i64 {
  %c = trait.coerce %v : i64 to i64
  return %c : i64
}
