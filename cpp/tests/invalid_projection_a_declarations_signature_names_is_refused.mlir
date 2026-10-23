// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s --check-prefix=ERASE

// A bodiless declaration's signature names a ground projection no impl
// resolves, and nothing calls it. The type its function-type attribute carries
// is a standing site, so the projection is work instantiation owes: it puts
// the projection to selection and, refused, names it at the declaration.
// Handed to erasure alone, the work is still owed, and erasure's entry refuses
// it there before anything else runs.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }

// ERASE: error: unresolved projection '!trait.proj<@A[i64], "Item">' before erasure
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i64], "Item">'}}
// expected-error @below {{unresolved projection '!trait.proj<@A[i64], "Item">' after instantiate-monomorphs}}
func.func private @ext(%x: !trait.proj<@A[i64], "Item">) -> i64

func.func @f() -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}
