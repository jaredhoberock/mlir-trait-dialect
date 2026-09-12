// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' | FileCheck %s

// The sibling of the refusal row: once @Gen binds @Gen[i64]::A to i64, the
// claim spelled through the projection and the claim spelled at its resolution
// are one claim. A witness for either verifies against @Box_i64, and impl
// selection serves an allege spelled through the projection from the same impl.

trait.trait private @Gen[!trait.poly<0>] {
  trait.assoc_type @A
}

trait.impl private @Gen_i64 for @Gen[i64] {
  trait.assoc_type @A = i64
}

trait.trait private @Box[!trait.poly<1>] {}

trait.impl private @Box_i64 for @Box[i64] {}

func.func private @holds(!trait.claim<@Box[i64]>,
                         !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>,
                         !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>)

// The witness spelled at the resolution, the witness spelled through the
// projection, and the allege selection serves all stand as one witness of
// @Box[i64].
// CHECK-LABEL: func.func @main
// CHECK: trait.witness @Box_i64 for @Box[i64]
// CHECK: trait.witness @Box_i64 for @Box[i64]
// CHECK: trait.witness @Box_i64 for @Box[i64]
// CHECK-NOT: trait.allege
func.func @main() {
  %direct = trait.witness @Box_i64 for @Box[i64]
  %through = trait.witness @Box_i64 for @Box[!trait.proj<@Gen[i64], "A">]
  %selected = trait.allege @Box[!trait.proj<@Gen[i64], "A">]
  trait.func.call @holds(%direct, %through, %selected)
    : (!trait.claim<@Box[i64] by @Box_i64>,
       !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i64>,
       !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>) -> ()
  return
}
