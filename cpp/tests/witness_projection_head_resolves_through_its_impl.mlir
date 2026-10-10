// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// The sibling of the refusal row: once @Gen binds @Gen[i64]::A to i64, the
// claim spelled through the projection and the claim spelled at its resolution
// are one claim. The witness of @Box_i64 is respelled through the projection
// by a coercion citing the binding, and impl selection serves an allege
// spelled through the projection from the same impl.

trait.trait private @Gen(%self: !trait.claim<@Gen[!trait.poly<0>]>) {
  trait.assoc_type @A
}

trait.impl private @Gen_i64(%self: !trait.claim<@Gen[i64]>) {
  trait.assoc_type @A = i64
}

trait.trait private @Box(%self: !trait.claim<@Box[!trait.poly<1>]>) {}

trait.impl private @Box_i64(%self: !trait.claim<@Box[i64]>) {}

func.func private @holds(!trait.claim<@Box[i64]>,
                         !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>,
                         !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>)

// The coercion settles to the witness of @Box_i64 respelled through the
// projection, and the allege selection serves at that spelling names the same
// proof: one application under one spelling names one proof.
// CHECK-LABEL: func.func @main
// CHECK: %[[D:.*]] = trait.witness @Box_i64 for @Box[i64]
// CHECK: %[[T:.*]] = trait.witness @[[P:Box_i64_.*]] for @Box[!trait.proj<@Gen[i64], "A">]
// CHECK: %[[S:.*]] = trait.witness @[[P]] for @Box[!trait.proj<@Gen[i64], "A">]
// CHECK: call @holds(%[[D]], %[[T]], %[[S]])
// CHECK-NOT: trait.allege
func.func @main() {
  %direct = trait.witness @Box_i64 for @Box[i64]
  %a = trait.witness proj_resolve !trait.proj<@Gen[i64], "A"> resolves i64 by @Gen_i64 : !trait.claim<!trait.proj<@Gen[i64], "A"> = i64>
  %through = trait.coerce %direct : !trait.claim<@Box[i64] by @Box_i64> to !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]> via (%a) : (!trait.claim<!trait.proj<@Gen[i64], "A"> = i64>)
  %selected = trait.allege @Box[!trait.proj<@Gen[i64], "A">]
  trait.func.call @holds(%direct, %through, %selected)
    : (!trait.claim<@Box[i64] by @Box_i64>,
       !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>,
       !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>) -> ()
  return
}
