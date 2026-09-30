// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --check-prefix=LOWERED

// An allegation over a generic associated type whose argument is a projection
// over a type variable is kept by the template, and the instance cut at i64
// proves it one step at a time: @Gat_any's binding at the argument as spelled,
// then the argument's own resolution through @Arg_i64, composed.

!S = !trait.poly<0>
!A = !trait.poly<1>
!T = !trait.poly<2>
!B = !trait.poly<3>
!X = !trait.poly<4>

trait.trait private @Arg[!S] { trait.assoc_type @Out }
trait.impl private @Arg_i64 for @Arg[i64] {
  trait.assoc_type @Out = i32
}

trait.trait private @Gat[!S] { trait.assoc_type @Item<[!A]> }
trait.impl private @Gat_any for @Gat[!T] {
  trait.assoc_type @Item<[!B]> = !B
}

func.func private @sink(%e: !trait.claim<!trait.proj<@Gat[!X], "Item", [!trait.proj<@Arg[!X], "Out">]> = i32>) {
  return
}

func.func private @send(%x: !X) {
  %e = trait.allege !trait.proj<@Gat[!X], "Item", [!trait.proj<@Arg[!X], "Out">]> = i32
  trait.func.call @sink(%e) : (!trait.claim<!trait.proj<@Gat[!X], "Item", [!trait.proj<@Arg[!X], "Out">]> = i32>) -> ()
  return
}

func.func @main() {
  %v = arith.constant 7 : i64
  trait.func.call @send(%v) : (i64) -> ()
  return
}

// CHECK-LABEL: func.func private @send(
// CHECK: trait.allege !trait.proj<@Gat[!trait.poly<4>], "Item", [!trait.proj<@Arg[!trait.poly<4>], "Out">]> = i32
// CHECK-LABEL: func.func private @send_
// CHECK-NOT: trait.allege
// CHECK: %[[GAT:.*]] = trait.witness proj_resolve !trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> resolves !trait.proj<@Arg[i64], "Out"> by @Gat_any[!trait.poly<2> = i64]
// CHECK: %[[ARG:.*]] = trait.witness proj_resolve !trait.proj<@Arg[i64], "Out"> resolves i32 by @Arg_i64
// CHECK: %[[BOTH:.*]] = trait.witness compose(%[[GAT]], %[[ARG]])
// CHECK: call @sink_{{.*}}(%[[BOTH]])

// LOWERED-NOT: trait.
// LOWERED-LABEL: func.func @main
