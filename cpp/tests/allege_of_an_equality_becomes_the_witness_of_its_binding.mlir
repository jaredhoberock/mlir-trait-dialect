// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --check-prefix=LOWERED

// An allegation of an equality, the type an impl binds a projection to, is
// proved by resolving the projection through the impl selection chooses for
// its application: it becomes the projection-resolution witness citing that
// impl at the arguments its parameters take there, with a witness of the
// premise the impl states.

!S = !trait.poly<0>
!T = !trait.poly<1>
!G = !trait.poly<2>

trait.trait private @Group[!S] {}
trait.impl private @Group_i8 for @Group[i8] {}

trait.trait private @Carry[!S, !G] { trait.assoc_type @Payload }
trait.impl private @Carry_any for @Carry[!T, !G] where [@Group[!G]] {
  trait.assoc_type @Payload = !T
}

func.func private @need(!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i64>)

func.func @main() {
  %e = trait.allege !trait.proj<@Carry[i64, i8], "Payload"> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i64>) -> ()
  return
}

// ROUNDTRIP: trait.allege !trait.proj<@Carry[i64, i8], "Payload"> = i64

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.allege
// CHECK: %[[GROUP:.*]] = trait.witness @Group_i8 for @Group[i8]
// CHECK: %[[PAYLOAD:.*]] = trait.witness proj_resolve !trait.proj<@Carry[i64, i8], "Payload"> resolves i64 by @Carry_any[!trait.poly<1> = i64, !trait.poly<2> = i8] given(%[[GROUP]])
// CHECK: call @need(%[[PAYLOAD]])

// LOWERED-LABEL: func.func @main
// LOWERED-NOT: trait.
// LOWERED: call @need()
