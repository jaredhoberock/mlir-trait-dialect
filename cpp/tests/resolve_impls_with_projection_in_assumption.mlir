// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

// Tests that projections in impl assumptions (where-clauses) are resolved
// before checking satisfiability. Without projection normalization, the
// assumption @Marker[!trait.proj<@Trait[i32], "Assoc">] would remain
// unresolved and fail to find the @Marker[()] impl.

!T = !trait.poly<0>

trait.trait @Trait[!T] {
  trait.assoc_type @Assoc
  func.func private @dummy(!T) -> i32
}

trait.trait @Marker[!T] {
  func.func private @mark(!T) -> i32
}

// impl Trait for i32 { type Assoc = tuple<>; }
trait.impl @Trait_i32 for @Trait[i32] {
  trait.assoc_type @Assoc = tuple<>
  func.func @dummy(%arg: i32) -> i32 {
    return %arg : i32
  }
}

// impl Marker for tuple<> {}
trait.impl @Marker_unit for @Marker[tuple<>] {
  func.func @mark(%arg: tuple<>) -> i32 {
    %c = arith.constant 1 : i32
    return %c : i32
  }
}

// impl<T: Trait> Marker for T where T::Assoc: Marker {}
!U = !trait.poly<1>
trait.impl @Marker_via_assoc for @Marker[!U] where [@Trait[!U], @Marker[!trait.proj<@Trait[!U], "Assoc">]] {
  func.func @mark(%arg: !U) -> i32 {
    %c = arith.constant 2 : i32
    return %c : i32
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

// CHECK: trait.proof @Marker_via_assoc_{{.*}}_p proves @Marker_via_assoc for @Marker[i32] given [@Trait_i32, @Marker_unit]
