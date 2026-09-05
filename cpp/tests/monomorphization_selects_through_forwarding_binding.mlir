// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Resolving a projection substitutes the selected impl's associated-type
// binding, and that binding may itself be a projection. Selection reaches the
// normal form of the demanded spelling, so a demand spelled through a
// forwarding impl selects the same impl the resolved spelling does. Reducing
// only one hop would leave the demand one hop short of the head the blanket
// impl is spelled against, and head matching would find no candidate.

!T = !trait.poly<0>
trait.trait private @Ten[!T] {
  trait.assoc_type @Element
}

trait.impl private @Ten_base for @Ten[i64] {
  trait.assoc_type @Element = i32
}

// The view forwards its element through its base.
trait.impl private @Ten_view for @Ten[f32] {
  trait.assoc_type @Element = !trait.proj<@Ten[i64], "Element">
}

!S = !trait.poly<1>
!O = !trait.poly<2>
trait.trait private @Get[!S, !O] {
  func.func nested @use()
}

!U = !trait.poly<3>
trait.impl private @Get_blanket for @Get[!U, !trait.proj<@Ten[!U], "Element">] where [@Ten[!U]] {
  func.func nested @use() {
    return
  }
}

// CHECK-LABEL: func.func @forwarded
// CHECK: call @Get_blanket
func.func @forwarded() {
  %c = trait.allege @Get[f32, !trait.proj<@Ten[f32], "Element">]
  trait.method.call %c @Get[f32, !trait.proj<@Ten[f32], "Element">]::@use() : () -> ()
  return
}
