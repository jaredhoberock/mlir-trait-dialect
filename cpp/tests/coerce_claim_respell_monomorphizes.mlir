// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A claim-respell coerce and its value-level twin, both consumed downstream,
// fold away at monomorphization: the respelled claim is used at its respelled
// application (a method call), and the value twin respells the ground value to
// the projection spelling. With every projection resolving to ground, the
// coerces and their cited equalities reduce to nothing and the surviving call is
// the concrete method. The respell sweep does not trip.

trait.trait @Bound[!trait.poly<0>] {
  func.func private @use(!trait.poly<0>) -> i1
}
trait.impl @Bound_impl for @Bound[i64] {
  func.func @use(%self: i64) -> i1 {
    %t = arith.constant 1 : i1
    return %t : i1
  }
}

trait.trait @Assoc[!trait.poly<0>] {
  trait.assoc_type @Output
}
trait.impl @Assoc_impl for @Assoc[i64] {
  trait.assoc_type @Output = i64
}

// CHECK-LABEL: func.func @respell_then_consume
// CHECK-NOT: trait.coerce
// CHECK: call @Bound_impl_use
func.func @respell_then_consume(%x: i64) -> i1 {
  %b = trait.allege @Bound[i64]
  %eq = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Output"> resolves i64 by @Assoc_impl
    : !trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[i64]>
    to !trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>
    via (%eq) : (!trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>)
  %vx = trait.coerce %x : i64 to !trait.proj<@Assoc[i64], "Output">
    via (%eq) : (!trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>)
  %r = trait.method.call %c @Bound[!trait.proj<@Assoc[i64], "Output">]::@use(%vx)
    : (!trait.proj<@Assoc[i64], "Output">) -> i1
  return %r : i1
}
