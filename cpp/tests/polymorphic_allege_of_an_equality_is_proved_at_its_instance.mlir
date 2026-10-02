// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --check-prefix=LOWERED

// An allegation of an equality over a type variable is kept by the template,
// as an allegation of an application is, and the instance cut at i64 proves it
// through the impl selection chooses for @Carry[i64, i8].

!S = !trait.poly<0>
!T = !trait.poly<1>
!G = !trait.poly<2>
!X = !trait.poly<3>

trait.trait private @Group(%self: !trait.claim<@Group[!S]>) {}
trait.impl private @Group_i8(%self: !trait.claim<@Group[i8]>) {}

trait.trait private @Carry(%self: !trait.claim<@Carry[!S, !G]>) { trait.assoc_type @Payload }
trait.impl private @Carry_any(%self: !trait.claim<@Carry[!T, !G]>, %group: !trait.claim<@Group[!G]>) {
  trait.assoc_type @Payload = !T
}

func.func private @sink(%e: !trait.claim<!trait.proj<@Carry[!X, i8], "Payload"> = !X>) {
  return
}

func.func private @send(%x: !X) {
  %e = trait.allege !trait.proj<@Carry[!X, i8], "Payload"> = !X
  trait.func.call @sink(%e) : (!trait.claim<!trait.proj<@Carry[!X, i8], "Payload"> = !X>) -> ()
  return
}

func.func @main() {
  %v = arith.constant 7 : i64
  trait.func.call @send(%v) : (i64) -> ()
  return
}

// CHECK-LABEL: func.func private @send(
// CHECK: trait.allege !trait.proj<@Carry[!trait.poly<3>, i8], "Payload"> = !trait.poly<3>
// CHECK-LABEL: func.func private @send_
// CHECK-NOT: trait.allege
// CHECK: trait.witness proj_resolve !trait.proj<@Carry[i64, i8], "Payload"> resolves i64 by @Carry_any given

// LOWERED-NOT: trait.
// LOWERED-LABEL: func.func @main
