// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" 2>&1 | FileCheck %s

// A binding terminal that still carries a projection is a promise like any
// other: the two spellings will denote one type once every lookup in them
// grounds. When monomorphization grounds them apart -- @Base[i64]::A to i32
// while tuple<@Base[i64]::B> grounds to tuple<i1> -- the coerce's ground
// endpoints stand apart and cannot cross the erase barrier every marked coerce
// must pass before LLVM; under mlir-opt's default verify-each the op verifier
// reports that same inconsistency the moment the pass grounds it. Admitting the
// shape at birth costs no tolerance at discharge.

trait.trait private @Base[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.impl private @Base_i64 for @Base[i64] {
  trait.assoc_type @A = i32
  trait.assoc_type @B = i1
}

// CHECK: error: 'trait.coerce' op input type 'i32' and result type 'tuple<i1>' are not consistent as a pending coerce
func.func @use(%x: !trait.proj<@Base[i64], "A">)
    -> tuple<!trait.proj<@Base[i64], "B">> {
  %y = trait.coerce %x : !trait.proj<@Base[i64], "A">
    to tuple<!trait.proj<@Base[i64], "B">> unproven
  return %y : tuple<!trait.proj<@Base[i64], "B">>
}
