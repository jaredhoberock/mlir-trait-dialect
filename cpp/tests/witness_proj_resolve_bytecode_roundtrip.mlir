// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --emit-bytecode | mlir-opt | FileCheck %s

// A proj-resolve witness carries an equality whose endpoints are types, in the
// attribute and in the equality-claim result alike. This pins that the two
// survive a bytecode round-trip unchanged and still agree.

!S = !trait.poly<0>

trait.trait private @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl private @Trait_impl for @Trait[i64] {
  trait.assoc_type @Output = i64
}

// CHECK: trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
func.func @resolve() -> !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
}
