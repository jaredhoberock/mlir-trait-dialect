// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s --check-prefix=ERASE

// An allegation that a projection no impl serves equals itself. The projection
// names no type, so neither side of the equality does, and selection resolves
// identical sides as it resolves any others: instantiation refuses to prove the
// allegation, naming why @Has[i32] has no impl, rather than read two identical
// spellings as settled. Erasure alone meets the allegation still owed to
// instantiation and refuses it there.

trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }

func.func @main() -> i64 {
  // ERASE: error: unproven monomorphic claim '!trait.claim<!trait.proj<@Has[i32], "Out"> = !trait.proj<@Has[i32], "Out">>' before erasure
  // expected-error @below {{'trait.allege' op no impl with satisfiable assumptions for '!trait.proj<@Has[i32], "Out">'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Has[i32], "Out"> = !trait.proj<@Has[i32], "Out">>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Has[i32], "Out"> = !trait.proj<@Has[i32], "Out">
  %k = arith.constant 7 : i64
  return %k : i64
}
