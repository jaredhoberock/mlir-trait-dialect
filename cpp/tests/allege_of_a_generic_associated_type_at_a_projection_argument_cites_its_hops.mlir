// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --check-prefix=LOWERED

// An allegation over a generic associated type whose argument is itself a
// projection is proved one step at a time, the argument first: the argument's
// own resolution, then the generic associated type's binding at the
// argument's normal form, which is that normal form. A coerce of the refl
// witness of the alleged equality's left side cites both steps to reach the
// alleged equality, its congruence carrying the projection as spelled to the
// one the second step states.

!S = !trait.poly<0>
!A = !trait.poly<1>

trait.trait private @Arg(%self: !trait.claim<@Arg[!S]>) {
  trait.assoc_type @Out
}
trait.impl private @Arg_i64(%self: !trait.claim<@Arg[i64]>) {
  trait.assoc_type @Out = i32
}

trait.trait private @Gat(%self: !trait.claim<@Gat[!S]>) {
  trait.assoc_type @Item<[!A]>
}
trait.impl private @Gat_i64(%self: !trait.claim<@Gat[i64]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.poly<0>
}

func.func private @need(!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = i32>)

func.func @main() {
  %e = trait.allege !trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = i32
  func.call @need(%e) : (!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = i32>) -> ()
  return
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.allege
// CHECK: %[[ARG:.*]] = trait.witness proj_resolve !trait.proj<@Arg[i64], "Out"> resolves i32 by @Arg_i64
// CHECK: %[[GAT:.*]] = trait.witness proj_resolve !trait.proj<@Gat[i64], "Item", [i32]> resolves i32 by @Gat_i64
// CHECK: %[[LHS:.*]] = trait.witness refl : !trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = !trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]>>
// CHECK: %[[BOTH:.*]] = trait.coerce %[[LHS]] : {{.*}} to !trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = i32> via (%[[ARG]], %[[GAT]])
// CHECK: call @need(%[[BOTH]])

// LOWERED-LABEL: func.func @main
// LOWERED-NOT: trait.
// LOWERED: call @need()
