// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" 2>&1 | FileCheck %s

// A bare-alias marked coerce is a promise that its two lookups denote one type.
// When monomorphization grounds them to DIFFERENT types the promise is broken:
// @Base[i64]::A resolves to i1, @Base[i64]::B to i32, and the coerce verifier
// reports the falsified obligation at the op once the module verifies. The
// marker never tolerates a survivor that ground to a lie.

trait.trait @Base[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.impl @Base_i64 for @Base[i64] {
  trait.assoc_type @A = i1
  trait.assoc_type @B = i32
}

// CHECK: error: 'trait.coerce' op input type 'i1' and result type 'i32' are not consistent as a pending coerce
func.func @use(%x: !trait.proj<@Base[i64], "A">) -> !trait.proj<@Base[i64], "B"> {
  %y = trait.coerce %x : !trait.proj<@Base[i64], "A">
    to !trait.proj<@Base[i64], "B"> unproven
  return %y : !trait.proj<@Base[i64], "B">
}
