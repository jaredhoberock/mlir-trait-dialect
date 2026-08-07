// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// trait.witness introduces equality evidence. The proj-resolve leaf cites an
// impl and freezes the equality it establishes; verifySymbolUses audits at the
// symbol seam that the impl binds @Trait[i64]::Output to i64, and verify()
// checks the result endpoints are an instance of the frozen certificate. The
// refl leaf introduces A = A with no citation.

!S = !trait.poly<0>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Output = i64
}

// CHECK-LABEL: func.func @resolve
// CHECK: trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
func.func @resolve() -> !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
}

// CHECK-LABEL: func.func @reflexive
// CHECK: trait.witness refl : !trait.claim<i64 = i64>
func.func @reflexive() -> !trait.claim<i64 = i64> {
  %r = trait.witness refl : !trait.claim<i64 = i64>
  return %r : !trait.claim<i64 = i64>
}
