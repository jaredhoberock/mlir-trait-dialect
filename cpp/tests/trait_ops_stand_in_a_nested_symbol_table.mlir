// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A trait op's parent must be a symbol table -- the requirement the Symbol trait
// itself verifies -- and nothing narrower: an impl stands inside a nested symbol
// table that is not a builtin.module. The names an impl resolves are anchored
// at the enclosing module, so the trait it implements is found there.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Tr(%self: !trait.claim<@Tr[!trait.poly<0>]>) {
  trait.method @m(!trait.poly<0>) -> i64
}

// CHECK: gpu.module @nested
// CHECK: trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>)
gpu.module @nested {
  trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
    trait.method @m(%x: i64) -> i64 { trait.return %x : i64 }
  }
}
