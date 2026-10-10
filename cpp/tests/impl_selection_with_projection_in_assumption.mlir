// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

// Tests that projections in impl assumptions (where-clauses) are resolved
// before checking satisfiability. Without projection normalization, the
// assumption @Marker[!trait.proj<@Trait[i32], "Assoc">] would remain
// unresolved and fail to find the @Marker[()] impl.

!T = !trait.poly<0>

trait.trait private @Trait(%self: !trait.claim<@Trait[!T]>) {
  trait.assoc_type @Assoc
  trait.method @dummy(!T) -> i32
}

trait.trait private @Marker(%self: !trait.claim<@Marker[!T]>) {
  trait.method @mark(!T) -> i32
}

// impl Trait for i32 { type Assoc = tuple<>; }
trait.impl private @Trait_i32(%self: !trait.claim<@Trait[i32]>) {
  trait.assoc_type @Assoc = tuple<>
  trait.method @dummy(%arg: i32) -> i32 {
    trait.return %arg : i32
  }
}

// impl Marker for tuple<> {}
trait.impl private @Marker_unit(%self: !trait.claim<@Marker[tuple<>]>) {
  trait.method @mark(%arg: tuple<>) -> i32 {
    %c = arith.constant 1 : i32
    trait.return %c : i32
  }
}

// impl<T: Trait> Marker for T where T::Assoc: Marker {}
!U = !trait.poly<1>
trait.impl private @Marker_via_assoc(%self: !trait.claim<@Marker[!U]>, %trait: !trait.claim<@Trait[!U]>, %marker: !trait.claim<@Marker[!trait.proj<@Trait[!U], "Assoc">]>) {
  trait.method @mark(%arg: !U) -> i32 {
    %c = arith.constant 2 : i32
    trait.return %c : i32
  }
}

// CHECK-LABEL: func.func @test
func.func @test() -> i32 {
  %x = arith.constant 42 : i32
  // Resolving @Marker[i32] should find @Marker_via_assoc, which requires:
  //   1. @Trait[i32] — satisfied by @Trait_i32
  //   2. @Marker[Trait[i32]::Assoc] — projection resolves to tuple<>,
  //      then @Marker[tuple<>] is satisfied by @Marker_unit
  // CHECK: trait.witness @Marker_via_assoc_{{.*}}_p for @Marker[i32]
  %m = trait.allege @Marker[i32]
  %res = trait.method.call %m @Marker[i32]::@mark(%x) : (i32) -> i32
  return %res : i32
}

// The proof witnesses @Marker_unit at the application it proves and coerces
// it to the where entry's spelling by the entry's projection resolved.
// CHECK: trait.proof private @Marker_via_assoc_{{.*}}_p {
// CHECK-NEXT: %[[T:.*]] = trait.witness @Trait_i32 for @Trait[i32]
// CHECK-NEXT: %[[U:.*]] = trait.witness @Marker_unit for @Marker[tuple<>]
// CHECK-NEXT: %[[E:.*]] = trait.witness proj_resolve !trait.proj<@Trait[i32], "Assoc"> resolves tuple<> by @Trait_i32
// CHECK-NEXT: %[[M:.*]] = trait.coerce %[[U]] : {{.*}} to !trait.claim<@Marker[!trait.proj<@Trait[i32], "Assoc">] by @Marker_unit> via (%[[E]])
// CHECK-NEXT: trait.derive @Marker[i32] from @Marker_via_assoc given(%[[T]], %[[M]])
