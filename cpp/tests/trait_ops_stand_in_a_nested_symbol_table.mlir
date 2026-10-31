// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A trait op's parent must be a symbol table -- the requirement the Symbol trait
// itself verifies -- and nothing narrower: a trait, an impl, their member
// functions and a call of a member stand inside a nested symbol table that is
// not a builtin.module. Every name they spell resolves in that table, as a
// func.call's callee does, so the program verifies and monomorphizes there:
// the instance is cut beside its function in the gpu.module, and the lowered
// func.call names it there.

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// VERIFIED: gpu.module @nested
// VERIFIED: trait.trait private @Tr(!trait.claim<@Tr[!trait.poly<0>]>) members [@Tr.m]
// VERIFIED: trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) members [@Tr.m -> @Tr_i64.m]

// CHECK: gpu.module @nested
// CHECK: func.func private @[[M:Tr_i64.m_h[0-9a-f]+]](%{{.*}}: i64) -> i64
// CHECK: func.func @use(%[[X:.*]]: i64) -> i64
// CHECK: call @[[M]](%[[X]])
// CHECK-NOT: trait.
gpu.module @nested {
  trait.trait private @Tr(!trait.claim<@Tr[!trait.poly<0>]>) members [@Tr.m]
  func.func private @Tr.m(!trait.claim<@Tr[!trait.poly<0>]>, !trait.poly<0>) -> i64
  trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) members [@Tr.m -> @Tr_i64.m] {
  }
  func.func private @Tr_i64.m(%self: !trait.claim<@Tr[i64]>, %x: i64) -> i64 {
    return %x : i64
  }
  func.func @use(%x: i64) -> i64 {
    %c = trait.allege @Tr[i64]
    %r = trait.func.call @Tr.m(%c, %x) : (!trait.claim<@Tr[i64]>, i64) -> i64
    return %r : i64
  }
}
