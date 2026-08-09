// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait,instantiate-monomorphs-trait)" | FileCheck %s

// The reconciliation walk runs at each instantiate epilogue by construction, so
// a second run re-drifts the bridged operand and re-mints the bridge: the
// respell rounds resolve the coerce's projection endpoints back to ground, the
// reflexive coerce folds away, and the epilogue mints one fresh coerce. Two runs
// therefore leave exactly one bridge, and the output verifies -- the churn is a
// property of running the step twice, which the shipped pipeline never does.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T7 = !trait.poly<7>

trait.trait @HasPart[!S] {
  trait.assoc_type @Part
}

trait.impl @HasPart_i64 for @HasPart[i64] {
  trait.assoc_type @Part = f32
}

trait.trait @Other[!S] {}

trait.impl @Other_f32 for @Other[f32] {}

trait.trait @Tr[!S] {}

trait.impl @CondImpl for @Tr[!X] where [@Other[!trait.proj<@HasPart[i64], "Part">]] {}

// Exactly one bridge survives two runs, feeding the one surviving derive.
// CHECK-COUNT-1: trait.coerce
// CHECK-NOT: trait.coerce
// CHECK: trait.derive @Tr[!trait.poly<7>] from @CondImpl
func.func private @template(
  %op: !trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">]>
) -> !trait.claim<@Tr[!T7]> {
  %d = trait.derive @Tr[!T7] from @CondImpl given(%op)
    : (!trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">]>)
  return %d : !trait.claim<@Tr[!T7]>
}
