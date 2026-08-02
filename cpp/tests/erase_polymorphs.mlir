// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// erase-polymorphs-trait is the second half of monomorphization: it erases the
// trait and impl templates, the claim and projection types, the projection
// casts, and the witnesses that instantiate-monomorphs left standing over the
// monomorphs it proved. The two run as a pipeline with per-pass verification
// off, since the state between them is not guaranteed to verify in general. This
// row pins that the erase step removes the trait and impl templates instantiation
// leaves standing — the two CHECK-NOTs just below.

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait,erase-polymorphs-trait)' --verify-each=false %s | FileCheck %s

// Concrete -> projection -> concrete roundtrip, monomorphized in two steps.

!T = !trait.poly<0>

trait.trait @Base[!T] {
  trait.assoc_type @Assoc
}

trait.impl @Base_i64 for @Base[i64] {
  trait.assoc_type @Assoc = i1
}

// no trait or impl template survives the erase
// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-LABEL: func.func @cast_roundtrip
// and the function body carries no projection cast, witness, projection type,
// or claim type
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: trait.witness
// CHECK-NOT: !trait.proj
// CHECK-NOT: !trait.claim
// CHECK: return %{{.*}} : i1
func.func @cast_roundtrip() -> i1 {
  %v = arith.constant true
  %w = trait.witness @Base_i64 for @Base[i64]
  // cast concrete i1 up to projection type
  %up = trait.proj.cast %v, %w : i1 to !trait.proj<@Base[i64], "Assoc"> by !trait.claim<@Base[i64] by @Base_i64>
  // cast projection type back down to concrete i1
  %down = trait.proj.cast %up, %w : !trait.proj<@Base[i64], "Assoc"> to i1 by !trait.claim<@Base[i64] by @Base_i64>
  return %down : i1
}
