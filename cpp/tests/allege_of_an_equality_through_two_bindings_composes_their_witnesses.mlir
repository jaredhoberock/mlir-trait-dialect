// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --check-prefix=LOWERED

// An allegation of an equality whose projection resolves through a binding that
// itself spells a projection is proved one binding at a time: a
// projection-resolution witness per projection resolved, composed into the
// alleged equality.

!S = !trait.poly<0>
!T = !trait.poly<1>
!G = !trait.poly<2>

trait.trait private @Other(%self: !trait.claim<@Other[!S]>) { trait.assoc_type @Out }
trait.impl private @Other_i64(%self: !trait.claim<@Other[i64]>) {
  trait.assoc_type @Out = i32
}

trait.trait private @Carry(%self: !trait.claim<@Carry[!S, !G]>) { trait.assoc_type @Payload }
trait.impl private @Carry_any(%self: !trait.claim<@Carry[!T, !G]>) {
  trait.assoc_type @Payload = !trait.proj<@Other[!T], "Out">
}

func.func private @need(!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i32>)

func.func @main() {
  %e = trait.allege !trait.proj<@Carry[i64, i8], "Payload"> = i32
  func.call @need(%e) : (!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i32>) -> ()
  return
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.allege
// CHECK: %[[CARRY:.*]] = trait.witness proj_resolve !trait.proj<@Carry[i64, i8], "Payload"> resolves !trait.proj<@Other[i64], "Out"> by @Carry_any
// CHECK: %[[OTHER:.*]] = trait.witness proj_resolve !trait.proj<@Other[i64], "Out"> resolves i32 by @Other_i64
// CHECK: %[[BOTH:.*]] = trait.witness compose(%[[CARRY]], %[[OTHER]])
// CHECK: call @need(%[[BOTH]])

// LOWERED-LABEL: func.func @main
// LOWERED-NOT: trait.
// LOWERED: call @need()
