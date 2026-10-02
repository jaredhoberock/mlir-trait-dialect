// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A generic associated type's bound is a requirement quantified over the
// associated type's own parameter: `type A<X>: Marker where X: Marker`. The
// trait declares it as a bodiless evidence method taking the binder's premise,
// and each impl computes the evidence that it holds for every argument, one
// body of each form: an impl that holds everywhere, the binder's premise, the
// impl's own where-clause entry, an impl whose premise the binder's premise
// discharges, the impl itself, and the impl's resolution of an equality.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
!B = !trait.poly<3>

trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!B]>]>
}

trait.impl private @Marker_i1(%self: !trait.claim<@Marker[i1]>) {}
trait.impl private @Marker_wrap(%self: !trait.claim<@Marker[tuple<!P>]>, %marker: !trait.claim<@Marker[!P]>) {}

trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = i1
  trait.method @requirement_0(%p: !trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!B]>]> {
    %r = trait.derive @Marker[!trait.proj<@Has[i32], "A", [!B]>] from @Marker_i1 given()
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [!B]>]>
  }
}

trait.impl private @Has_i64(%self: !trait.claim<@Has[i64]>) {
  trait.assoc_type @A<[!X]> = !X
  trait.method @requirement_0(%p: !trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[i64], "A", [!B]>]> {
    %a = trait.witness proj_resolve !trait.proj<@Has[i64], "A", [!B]> resolves !B by @Has_i64 : !trait.claim<!trait.proj<@Has[i64], "A", [!B]> = !B>
    %r = trait.coerce %p : !trait.claim<@Marker[!B]> to !trait.claim<@Marker[!trait.proj<@Has[i64], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Has[i64], "A", [!B]> = !B>)
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Has[i64], "A", [!B]>]>
  }
}

trait.impl private @Has_tuple(%self: !trait.claim<@Has[tuple<!P>]>, %marker: !trait.claim<@Marker[!P]>) {
  trait.assoc_type @A<[!X]> = !P
  trait.method @requirement_0(%p: !trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[tuple<!P>], "A", [!B]>]> {
    %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!P>], "A", [!B]> resolves !P by @Has_tuple given(%marker) : (!trait.claim<@Marker[!P]>) : !trait.claim<!trait.proj<@Has[tuple<!P>], "A", [!B]> = !P>
    %r = trait.coerce %marker : !trait.claim<@Marker[!P]> to !trait.claim<@Marker[!trait.proj<@Has[tuple<!P>], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!P>], "A", [!B]> = !P>)
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Has[tuple<!P>], "A", [!B]>]>
  }
}

trait.impl private @Has_f32(%self: !trait.claim<@Has[f32]>) {
  trait.assoc_type @A<[!X]> = tuple<!X>
  trait.method @requirement_0(%p: !trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Has[f32], "A", [!B]>]> {
    %w = trait.derive @Marker[tuple<!B>] from @Marker_wrap given(%p) : (!trait.claim<@Marker[!B]>)
    %a = trait.witness proj_resolve !trait.proj<@Has[f32], "A", [!B]> resolves tuple<!B> by @Has_f32 : !trait.claim<!trait.proj<@Has[f32], "A", [!B]> = tuple<!B>>
    %r = trait.coerce %w : !trait.claim<@Marker[tuple<!B>]> to !trait.claim<@Marker[!trait.proj<@Has[f32], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Has[f32], "A", [!B]> = tuple<!B>>)
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Has[f32], "A", [!B]>]>
  }
}

trait.trait private @Self(%self: !trait.claim<@Self[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Self[!trait.proj<@Self[!S], "A", [!B]>]>
}
trait.impl private @Self_i32(%self: !trait.claim<@Self[i32]>) {
  trait.assoc_type @A<[!X]> = i32
  trait.method @requirement_0() -> !trait.claim<@Self[!trait.proj<@Self[i32], "A", [!B]>]> {
    %w = trait.witness @Self_i32 for @Self[i32]
    %a = trait.witness proj_resolve !trait.proj<@Self[i32], "A", [!B]> resolves i32 by @Self_i32 : !trait.claim<!trait.proj<@Self[i32], "A", [!B]> = i32>
    %r = trait.coerce %w : !trait.claim<@Self[i32] by @Self_i32> to !trait.claim<@Self[!trait.proj<@Self[i32], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Self[i32], "A", [!B]> = i32>)
    trait.return %r : !trait.claim<@Self[!trait.proj<@Self[i32], "A", [!B]>]>
  }
}

// An equality conclusion is the impl's own resolution of the projection when
// its binding makes the two sides one type.
trait.trait private @Same(%self: !trait.claim<@Same[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<!trait.proj<@Same[!S], "A", [!B]> = !B>
}
trait.impl private @Same_i32(%self: !trait.claim<@Same[i32]>) {
  trait.assoc_type @A<[!X]> = !X
  trait.method @requirement_0() -> !trait.claim<!trait.proj<@Same[i32], "A", [!B]> = !B> {
    %r = trait.witness proj_resolve !trait.proj<@Same[i32], "A", [!B]> resolves !B by @Same_i32 : !trait.claim<!trait.proj<@Same[i32], "A", [!B]> = !B>
    trait.return %r : !trait.claim<!trait.proj<@Same[i32], "A", [!B]> = !B>
  }
}

// CHECK: trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) {
// CHECK: trait.method @requirement_0(!trait.claim<@Marker[!trait.poly<3>]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<3>]>]>
// CHECK-LABEL: trait.impl private @Has_i32
// CHECK: trait.derive @Marker[!trait.proj<@Has[i32], "A", [!trait.poly<3>]>] from @Marker_i1 given()
// CHECK-LABEL: trait.impl private @Has_i64
// CHECK: trait.coerce %arg0 : !trait.claim<@Marker[!trait.poly<3>]>
// CHECK-LABEL: trait.impl private @Has_tuple
// CHECK: trait.coerce %marker : !trait.claim<@Marker[!trait.poly<2>]>
// CHECK-LABEL: trait.impl private @Has_f32
// CHECK: trait.derive @Marker[tuple<!trait.poly<3>>] from @Marker_wrap given(%arg0)
// CHECK-LABEL: trait.impl private @Self_i32
// CHECK: trait.witness @Self_i32 for @Self[i32]
// CHECK-LABEL: trait.impl private @Same_i32
// CHECK: trait.witness proj_resolve !trait.proj<@Same[i32], "A", [!trait.poly<3>]> resolves !trait.poly<3> by @Same_i32
