// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// erase-polymorphs-trait is the second half of monomorphization: it erases the
// claim and projection types, the coerces, and the witnesses that
// instantiate-monomorphs left standing over the monomorphs it proved, and holds
// what stands outside a template theory-free. It deletes no template; the
// collector it runs after those erasures takes the trait and impl templates
// nothing names — the two CHECK-NOTs just below.

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait,erase-polymorphs-trait)' %s | FileCheck %s

// Concrete -> projection -> concrete roundtrip, monomorphized in two steps.

!T = !trait.poly<0>

trait.trait private @Base[!T] {
  trait.assoc_type @Assoc
}

trait.impl private @Base_i64 for @Base[i64] {
  trait.assoc_type @Assoc = i1
}

// no trait or impl template survives the erase
// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-LABEL: func.func @cast_roundtrip
// and the function body carries no coerce, witness, projection type, or claim
// type
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.witness
// CHECK-NOT: !trait.proj
// CHECK-NOT: !trait.claim
// CHECK: return %{{.*}} : i1
func.func @cast_roundtrip() -> i1 {
  %v = arith.constant true
  %e = trait.witness proj_resolve !trait.proj<@Base[i64], "Assoc"> resolves i1 by @Base_i64
    : !trait.claim<!trait.proj<@Base[i64], "Assoc"> = i1>
  // coerce concrete i1 up to projection type
  %up = trait.coerce %v : i1 to !trait.proj<@Base[i64], "Assoc"> via (%e) : (!trait.claim<!trait.proj<@Base[i64], "Assoc"> = i1>)
  // coerce projection type back down to concrete i1
  %down = trait.coerce %up : !trait.proj<@Base[i64], "Assoc"> to i1 via (%e) : (!trait.claim<!trait.proj<@Base[i64], "Assoc"> = i1>)
  return %down : i1
}
