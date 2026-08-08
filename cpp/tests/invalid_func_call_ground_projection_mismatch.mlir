// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A func.call verifies its signature with the module-free comparator: each
// declared formal is unified against the actual with no module to resolve a
// projection. A ground-projection formal meeting a mismatched concrete actual
// has no spelling the comparator can equate, so the crossing is rejected.

!S = !trait.poly<0>

trait.trait @T[!S] {
  trait.assoc_type @Out
}

func.func @callee(%p: !trait.proj<@T[i64], "Out">) -> !trait.proj<@T[i64], "Out"> {
  return %p : !trait.proj<@T[i64], "Out">
}

func.func @caller(%x: i32) -> !trait.proj<@T[i64], "Out"> {
  // expected-error @below {{projection mismatch: expected '!trait.proj<@T[i64], "Out">' but found 'i32'}}
  %r = trait.func.call @callee(%x) : (i32) -> !trait.proj<@T[i64], "Out">
  return %r : !trait.proj<@T[i64], "Out">
}
