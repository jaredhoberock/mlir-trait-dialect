// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// `type A<X>: Marker where X: Marker`, called at an argument for X with a
// claim of the premise there: a generic function reaches Marker of `T::A<i1>`
// through its `Has` claim's evidence member, and a default reaches it
// through its trait's self claim, which the impl leaves its member to. Both
// instances lower to the member function the impl binding's type implements.

// VERIFIED: func.func private @Has.use(%[[SELF:[^:]*]]: !trait.claim<@Has[!trait.poly<0>]>, %{{[^:]*}}: {{.*}}, %[[P:[^:]*]]: !trait.claim<@Marker[i1]>)
// VERIFIED-NEXT: trait.func.call @Has.requirement_0(%[[SELF]], %[[P]]) : (!trait.claim<@Has[!trait.poly<0>]>,
// VERIFIED: func.func private @f(
// VERIFIED-NEXT: trait.func.call @Has.requirement_0(%{{.*}}, %{{.*}}) : (!trait.claim<@Has[!trait.poly<0>]>,

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>
!B = !trait.poly<4>

trait.trait private @Marker(!trait.claim<@Marker[!S]>) members [@Marker.mark]
func.func private @Marker.mark(!trait.claim<@Marker[!S]>, !S) -> i64

trait.trait private @Has(!trait.claim<@Has[!S]>) assoc {A = 1} members [@Has.requirement_0, @Has.use]
func.func private @Has.requirement_0(!trait.claim<@Has[!S]>, !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
func.func private @Has.use(%self: !trait.claim<@Has[!S]>, %x: !trait.proj<@Has[!S], "A", [i1]>, %p: !trait.claim<@Marker[i1]>) -> i64 {
  %m = trait.func.call @Has.requirement_0(%self, %p) : (!trait.claim<@Has[!S]>, !trait.claim<@Marker[i1]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [i1]>]>
  %r = trait.func.call @Marker.mark(%m, %x) : (!trait.claim<@Marker[!trait.proj<@Has[!S], "A", [i1]>]>, !trait.proj<@Has[!S], "A", [i1]>) -> i64
  return %r : i64
}

trait.impl private @Marker_i1(%self: !trait.claim<@Marker[i1]>) members [@Marker.mark -> @Marker_i1.mark] {
}
func.func private @Marker_i1.mark(%self: !trait.claim<@Marker[i1]>, %x: i1) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) bindings {A = i1} members [@Has.requirement_0 -> @Has_i32.requirement_0] {
}
func.func private @Has_i32.requirement_0(%self: !trait.claim<@Has[i32]>, %p: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<0>]>]> {
  %w = trait.derive @Marker[i1] from @Marker_i1 given()
  %a = trait.witness proj_resolve !trait.proj<@Has[i32], "A", [!trait.poly<0>]> resolves i1 by @Has_i32 : !trait.claim<!trait.proj<@Has[i32], "A", [!trait.poly<0>]> = i1>
  %r = trait.coerce %w : !trait.claim<@Marker[i1]> to !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<0>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[i32], "A", [!trait.poly<0>]> = i1>)
  return %r : !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<0>]>]>
}

func.func private @use_marker(%m: !trait.claim<@Marker[!trait.poly<0>]>, %x: !trait.poly<0>) -> i64 {
  %r = trait.func.call @Marker.mark(%m, %x) : (!trait.claim<@Marker[!trait.poly<0>]>, !trait.poly<0>) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Has[!trait.poly<0>]>, %p: !trait.claim<@Marker[i1]>, %x: !trait.proj<@Has[!trait.poly<0>], "A", [i1]>) -> i64 {
  %m = trait.func.call @Has.requirement_0(%h, %p) : (!trait.claim<@Has[!trait.poly<0>]>, !trait.claim<@Marker[i1]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [i1]>]>
  %r = trait.func.call @use_marker(%m, %x) : (!trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [i1]>]>, !trait.proj<@Has[!trait.poly<0>], "A", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Has[i32], "A", [i1]>) -> i64 {
  %h = trait.allege @Has[i32]
  %p = trait.allege @Marker[i1]
  %r = trait.func.call @f(%h, %p, %x) : (!trait.claim<@Has[i32]>, !trait.claim<@Marker[i1]>, !trait.proj<@Has[i32], "A", [i1]>) -> i64
  %u = trait.func.call @Has.use(%h, %x, %p) : (!trait.claim<@Has[i32]>, !trait.proj<@Has[i32], "A", [i1]>, !trait.claim<@Marker[i1]>) -> i64
  %s = arith.addi %r, %u : i64
  return %s : i64
}

// CHECK-DAG: func.func private @[[MARK:Marker_i1.mark_h[0-9a-f]+]](%{{.*}}: i1) -> i64
// CHECK-DAG: func.func private @[[USE:Has.use_h[0-9a-f]+]](%{{.*}}: i1) -> i64
// CHECK-DAG: func.func private @[[F:f_[_a-z0-9]*]](%{{.*}}: i1) -> i64
// CHECK: func.func @main(%{{.*}}: i1) -> i64
// CHECK-DAG: call @[[F]](
// CHECK-DAG: call @[[USE]](
// CHECK-NOT: trait.
