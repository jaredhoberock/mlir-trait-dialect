// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait with two blanket impls is one the module alone answers nothing about:
// both headers bind every application, and which of them serves one is decided
// by their where clauses, which only impl selection weighs. So @Takes_blanket's
// header, spelling @Item[Self]::Of, reaches the @Takes[i32, i64] demanded below
// only through what selection settled for @Item[i32] -- and every step that
// rebuilds that header, from the candidate match through the proof that names
// it and the instance that proof is named after, reads it that way.

!T = !trait.poly<0>
trait.trait private @Item[!T] {
  trait.assoc_type @Of
}

!S = !trait.poly<1>
trait.trait private @Small[!S] {
}

!B = !trait.poly<2>
trait.trait private @Big[!B] {
}

!A = !trait.poly<3>
trait.impl private @Item_small for @Item[!A] where [@Small[!A]] {
  trait.assoc_type @Of = i64
}

!C = !trait.poly<4>
trait.impl private @Item_big for @Item[!C] where [@Big[!C]] {
  trait.assoc_type @Of = f32
}

trait.impl private @Small_i32 for @Small[i32] {
}

!U = !trait.poly<5>
!V = !trait.poly<6>
trait.trait private @Takes[!U, !V] {
  func.func private @go(!U) -> !V
}

!W = !trait.poly<7>
trait.impl private @Takes_blanket
    for @Takes[!W, !trait.proj<@Item[!W], "Of">] where [@Item[!W]] {
  func.func @go(%x: !W) -> !trait.proj<@Item[!W], "Of"> {
    %r = ub.poison : !trait.proj<@Item[!W], "Of">
    return %r : !trait.proj<@Item[!W], "Of">
  }
}

// CHECK-LABEL: func.func private @Takes_blanket_{{[0-9a-z_]+}}_go
// CHECK-SAME: (%{{.*}}: i32) -> i64
// CHECK-LABEL: func.func @main
// CHECK: call @Takes_blanket_
func.func @main(%x: i32) -> !trait.proj<@Item[i32], "Of"> {
  %item = trait.allege @Item[i32]
  %takes = trait.derive @Takes[i32, !trait.proj<@Item[i32], "Of">]
    from @Takes_blanket given(%item) : (!trait.claim<@Item[i32]>)
  %r = trait.method.call %takes
    @Takes[i32, !trait.proj<@Item[i32], "Of">]::@go(%x)
    : (i32) -> !trait.proj<@Item[i32], "Of">
  return %r : !trait.proj<@Item[i32], "Of">
}
