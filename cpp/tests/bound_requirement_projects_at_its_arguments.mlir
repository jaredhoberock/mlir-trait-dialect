// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// `type A<X>: Marker where X: Marker`, called at an argument for X with a
// claim of the premise there: a generic function reaches Marker of `T::A<i1>`
// through its `Has` claim's evidence method, and a default method reaches it
// through its trait's self claim, which cloning carries into an impl that
// defines no method of its own. Both instances lower to the method the impl
// binding's type implements.

// VERIFIED: trait.method.call %self @Has[!trait.poly<0>]::@requirement_0(%arg1)
// VERIFIED: trait.method.call %{{.*}} @Has[!trait.poly<2>]::@requirement_0(%{{.*}})

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>
!B = !trait.poly<4>

trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {
  trait.method @mark(!S) -> i64
}

trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!B]>]>
  trait.method @use(%x: !trait.proj<@Has[!S], "A", [i1]>, %p: !trait.claim<@Marker[i1]>) -> i64 {
    %m = trait.method.call %self @Has[!S]::@requirement_0(%p) : (!trait.claim<@Marker[i1]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [i1]>]>
    %r = trait.method.call %m @Marker[!trait.proj<@Has[!S], "A", [i1]>]::@mark(%x) : (!trait.proj<@Has[!S], "A", [i1]>) -> i64
    trait.return %r : i64
  }
}

trait.impl private @Marker_i1(%self: !trait.claim<@Marker[i1]>) {
  trait.method @mark(%x: i1) -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}

trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = i1
  trait.method @requirement_0(%p: !trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!B]>]> {
    %r = trait.derive @Marker[!trait.proj<@Has[i32], "A", [!B]>] from @Marker_i1 given()
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!B]>]>
  }
}

func.func private @use_marker(%m: !trait.claim<@Marker[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Marker[!M]::@mark(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Has[!T]>, %p: !trait.claim<@Marker[i1]>, %x: !trait.proj<@Has[!T], "A", [i1]>) -> i64 {
  %m = trait.method.call %h @Has[!T]::@requirement_0(%p) : (!trait.claim<@Marker[i1]>) -> !trait.claim<@Marker[!trait.proj<@Has[!T], "A", [i1]>]>
  %r = trait.func.call @use_marker(%m, %x) : (!trait.claim<@Marker[!trait.proj<@Has[!T], "A", [i1]>]>, !trait.proj<@Has[!T], "A", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Has[i32], "A", [i1]>) -> i64 {
  %h = trait.allege @Has[i32]
  %p = trait.allege @Marker[i1]
  %r = trait.func.call @f(%h, %p, %x) : (!trait.claim<@Has[i32]>, !trait.claim<@Marker[i1]>, !trait.proj<@Has[i32], "A", [i1]>) -> i64
  %u = trait.method.call %h @Has[i32]::@use(%x, %p) : (!trait.proj<@Has[i32], "A", [i1]>, !trait.claim<@Marker[i1]>) -> i64
  %s = arith.addi %r, %u : i64
  return %s : i64
}

// CHECK-DAG: func.func private @[[MARK:Marker_i1_h[0-9a-f]+_mark]](%{{.*}}: i1) -> i64
// CHECK-DAG: func.func private @[[USE:Has_h[0-9a-f]+_use]](%{{.*}}: i1) -> i64
// CHECK-DAG: func.func private @[[F:f_[_a-z0-9]*]](%{{.*}}: i1) -> i64
// CHECK: func.func @main(%{{.*}}: i1) -> i64
// CHECK-DAG: call @[[F]](
// CHECK-DAG: call @[[USE]](
// CHECK-NOT: trait.
