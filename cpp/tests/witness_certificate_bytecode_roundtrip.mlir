// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --emit-bytecode | mlir-opt | FileCheck %s

// The certificate's endpoints are opaque to sub-element walking, so
// bytecode cannot recover them from the generic walk; the attribute's own
// print/parse carries them. This pins that a proj-resolve witness -- capsule
// and equality-claim result together -- survives a bytecode round-trip
// unchanged.

!S = !trait.poly<0>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Output = i64
}

// CHECK: trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
func.func @resolve() -> !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
}
