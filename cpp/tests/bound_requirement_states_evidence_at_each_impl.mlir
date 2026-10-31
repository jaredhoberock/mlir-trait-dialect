// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A generic associated type's bound is a requirement quantified over the
// associated type's own parameter: `type A<X>: Marker where X: Marker`. The
// trait declares it as a bodiless evidence member taking the binder's premise,
// and each impl computes the evidence that it holds for every argument, one
// body of each form: an impl that holds everywhere, the binder's premise, the
// impl's own where-clause entry, an impl whose premise the binder's premise
// discharges, the impl itself, and the impl's resolution of an equality.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
!B = !trait.poly<3>

trait.trait private @Marker(!trait.claim<@Marker[!S]>)
trait.trait private @Has(!trait.claim<@Has[!S]>) assoc {A = 1} members [@Has.requirement_0]
func.func private @Has.requirement_0(!trait.claim<@Has[!S]>, !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>

trait.impl private @Marker_i1(%self: !trait.claim<@Marker[i1]>) {
}
trait.impl private @Marker_wrap(%self: !trait.claim<@Marker[tuple<!trait.poly<0>>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) {
}

trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) bindings {A = i1} members [@Has.requirement_0 -> @Has_i32.requirement_0] {
}
func.func private @Has_i32.requirement_0(%self: !trait.claim<@Has[i32]>, %p: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<0>]>]> {
  %w = trait.derive @Marker[i1] from @Marker_i1 given()
  %a = trait.witness proj_resolve !trait.proj<@Has[i32], "A", [!trait.poly<0>]> resolves i1 by @Has_i32 : !trait.claim<!trait.proj<@Has[i32], "A", [!trait.poly<0>]> = i1>
  %r = trait.coerce %w : !trait.claim<@Marker[i1]> to !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<0>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[i32], "A", [!trait.poly<0>]> = i1>)
  return %r : !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<0>]>]>
}

trait.impl private @Has_i64(%self: !trait.claim<@Has[i64]>) bindings {A = !trait.poly<0>} members [@Has.requirement_0 -> @Has_i64.requirement_0] {
}
func.func private @Has_i64.requirement_0(%self: !trait.claim<@Has[i64]>, %p: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[!trait.proj<@Has[i64], "A", [!trait.poly<0>]>]> {
  %a = trait.witness proj_resolve !trait.proj<@Has[i64], "A", [!trait.poly<0>]> resolves !trait.poly<0> by @Has_i64 : !trait.claim<!trait.proj<@Has[i64], "A", [!trait.poly<0>]> = !trait.poly<0>>
  %r = trait.coerce %p : !trait.claim<@Marker[!trait.poly<0>]> to !trait.claim<@Marker[!trait.proj<@Has[i64], "A", [!trait.poly<0>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[i64], "A", [!trait.poly<0>]> = !trait.poly<0>>)
  return %r : !trait.claim<@Marker[!trait.proj<@Has[i64], "A", [!trait.poly<0>]>]>
}

trait.impl private @Has_tuple(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) bindings {A = !trait.poly<0>} members [@Has.requirement_0 -> @Has_tuple.requirement_0] {
}
func.func private @Has_tuple.requirement_0(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>, %p: !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
  %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.poly<0> by @Has_tuple[!trait.poly<0>] given(%marker) : (!trait.claim<@Marker[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>
  %r = trait.coerce %marker : !trait.claim<@Marker[!trait.poly<0>]> to !trait.claim<@Marker[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>)
  return %r : !trait.claim<@Marker[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
}

trait.impl private @Has_f32(%self: !trait.claim<@Has[f32]>) bindings {A = tuple<!trait.poly<0>>} members [@Has.requirement_0 -> @Has_f32.requirement_0] {
}
func.func private @Has_f32.requirement_0(%self: !trait.claim<@Has[f32]>, %p: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[!trait.proj<@Has[f32], "A", [!trait.poly<0>]>]> {
  %w = trait.derive @Marker[tuple<!trait.poly<0>>] from @Marker_wrap[!trait.poly<0>] given(%p) : (!trait.claim<@Marker[!trait.poly<0>]>)
  %a = trait.witness proj_resolve !trait.proj<@Has[f32], "A", [!trait.poly<0>]> resolves tuple<!trait.poly<0>> by @Has_f32 : !trait.claim<!trait.proj<@Has[f32], "A", [!trait.poly<0>]> = tuple<!trait.poly<0>>>
  %r = trait.coerce %w : !trait.claim<@Marker[tuple<!trait.poly<0>>]> to !trait.claim<@Marker[!trait.proj<@Has[f32], "A", [!trait.poly<0>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[f32], "A", [!trait.poly<0>]> = tuple<!trait.poly<0>>>)
  return %r : !trait.claim<@Marker[!trait.proj<@Has[f32], "A", [!trait.poly<0>]>]>
}

trait.trait private @Self(!trait.claim<@Self[!S]>) assoc {A = 1} members [@Self.requirement_0]
func.func private @Self.requirement_0(!trait.claim<@Self[!S]>) -> !trait.claim<@Self[!trait.proj<@Self[!trait.poly<0>], "A", [!trait.poly<1>]>]>
trait.impl private @Self_i32(%self: !trait.claim<@Self[i32]>) bindings {A = i32} members [@Self.requirement_0 -> @Self_i32.requirement_0] {
}
func.func private @Self_i32.requirement_0(%self: !trait.claim<@Self[i32]>) -> !trait.claim<@Self[!trait.proj<@Self[i32], "A", [!trait.poly<0>]>]> {
  %w = trait.witness @Self_i32 for @Self[i32]
  %a = trait.witness proj_resolve !trait.proj<@Self[i32], "A", [!trait.poly<0>]> resolves i32 by @Self_i32 : !trait.claim<!trait.proj<@Self[i32], "A", [!trait.poly<0>]> = i32>
  %r = trait.coerce %w : !trait.claim<@Self[i32] by @Self_i32> to !trait.claim<@Self[!trait.proj<@Self[i32], "A", [!trait.poly<0>]>]> via (%a) : (!trait.claim<!trait.proj<@Self[i32], "A", [!trait.poly<0>]> = i32>)
  return %r : !trait.claim<@Self[!trait.proj<@Self[i32], "A", [!trait.poly<0>]>]>
}

// An equality conclusion is the impl's own resolution of the projection when
// its binding makes the two sides one type.
trait.trait private @Same(!trait.claim<@Same[!S]>) assoc {A = 1} members [@Same.requirement_0]
func.func private @Same.requirement_0(!trait.claim<@Same[!S]>) -> !trait.claim<!trait.proj<@Same[!trait.poly<0>], "A", [!trait.poly<1>]> = !trait.poly<1>>
trait.impl private @Same_i32(%self: !trait.claim<@Same[i32]>) bindings {A = !trait.poly<0>} members [@Same.requirement_0 -> @Same_i32.requirement_0] {
}
func.func private @Same_i32.requirement_0(%self: !trait.claim<@Same[i32]>) -> !trait.claim<!trait.proj<@Same[i32], "A", [!trait.poly<0>]> = !trait.poly<0>> {
  %r = trait.witness proj_resolve !trait.proj<@Same[i32], "A", [!trait.poly<0>]> resolves !trait.poly<0> by @Same_i32 : !trait.claim<!trait.proj<@Same[i32], "A", [!trait.poly<0>]> = !trait.poly<0>>
  return %r : !trait.claim<!trait.proj<@Same[i32], "A", [!trait.poly<0>]> = !trait.poly<0>>
}

// CHECK: trait.trait private @Has(!trait.claim<@Has[!trait.poly<0>]>) assoc {A = 1} members [@Has.requirement_0]
// CHECK: func.func private @Has.requirement_0(!trait.claim<@Has[!trait.poly<0>]>, !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
// CHECK-LABEL: func.func private @Has_i32.requirement_0
// CHECK: trait.derive @Marker[i1] from @Marker_i1 given()
// CHECK-LABEL: func.func private @Has_i64.requirement_0
// CHECK-SAME: %[[PREMISE:[^:]*]]: !trait.claim<@Marker[!trait.poly<0>]>
// CHECK: trait.coerce %[[PREMISE]] : !trait.claim<@Marker[!trait.poly<0>]>
// CHECK-LABEL: func.func private @Has_tuple.requirement_0
// CHECK-SAME: %[[MARKER:[^:]*]]: !trait.claim<@Marker[!trait.poly<0>]>, %{{[^:]*}}: !trait.claim<@Marker[!trait.poly<1>]>
// CHECK: trait.coerce %[[MARKER]] : !trait.claim<@Marker[!trait.poly<0>]>
// CHECK-LABEL: func.func private @Has_f32.requirement_0
// CHECK-SAME: %[[P32:[^:]*]]: !trait.claim<@Marker[!trait.poly<0>]>
// CHECK: trait.derive @Marker[tuple<!trait.poly<0>>] from @Marker_wrap[!trait.poly<0>] given(%[[P32]])
// CHECK-LABEL: func.func private @Self_i32.requirement_0
// CHECK: trait.witness @Self_i32 for @Self[i32]
// CHECK-LABEL: func.func private @Same_i32.requirement_0
// CHECK: trait.witness proj_resolve !trait.proj<@Same[i32], "A", [!trait.poly<0>]> resolves !trait.poly<0> by @Same_i32
