// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A trait op's parent must be a symbol table -- the requirement the Symbol trait
// itself verifies -- and nothing narrower: a trait and an impl stand inside a
// nested symbol table that is not a builtin.module. The names an impl resolves
// are anchored at the enclosing module, so the trait it implements is found
// there.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Tr[!trait.poly<0>] {
  func.func private @m(!trait.poly<0>) -> i64
}

// CHECK: gpu.module @nested
// CHECK: trait.trait private @Inner
// CHECK: trait.impl private @Tr_i64 for @Tr[i64]
gpu.module @nested {
  trait.trait private @Inner[!trait.poly<0>] {
    func.func private @n(!trait.poly<0>) -> i64
  }

  trait.impl private @Tr_i64 for @Tr[i64] {
    func.func @m(%self: i64) -> i64 { return %self : i64 }
  }
}
