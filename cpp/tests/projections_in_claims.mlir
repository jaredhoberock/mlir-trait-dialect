// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests that projection types inside claim parameters are resolved during
// monomorphization. Without this, trait.assume ops can't match their
// corresponding proven claim parameters when the claim's type argument
// is a projection (e.g., @Id[proj<@Prod[i64], "Out">] must become @Id[i64]).

// --- Shared trait definitions ---

trait.trait @Prod[!trait.poly<0>] {
  trait.assoc_type @Out
  func.func nested @get(!trait.poly<0>) -> !trait.proj<@Prod[!trait.poly<0>], "Out">
}

trait.trait @Id[!trait.poly<1>] {
  func.func nested @id(!trait.poly<1>) -> !trait.poly<1>
}

trait.trait @Map[!trait.poly<4>] {
  trait.assoc_type @Result
  func.func nested @apply(!trait.poly<4>) -> !trait.proj<@Map[!trait.poly<4>], "Result">
}

// --- Shared impl definitions ---

trait.impl @Id_blanket for @Id[!trait.poly<5>] {
  func.func nested @id(%x: !trait.poly<5>) -> !trait.poly<5> {
    %0 = trait.assume @Id[!trait.poly<5>]
    return %x : !trait.poly<5>
  }
}

trait.impl @Prod_i64 for @Prod[i64] {
  trait.assoc_type @Out = i64
  func.func nested @get(%x: i64) -> i64 {
    %0 = trait.assume @Prod[i64]
    return %x : i64
  }
}

trait.impl @Map_i64 for @Map[i64] {
  trait.assoc_type @Result = i64
  func.func nested @apply(%x: i64) -> i64 {
    %0 = trait.assume @Map[i64]
    return %x : i64
  }
}

// --- Test 1: func.call with projection in claim + trait.assume ---
// The claim @Id[proj<@Prod[T], "Out">] must be resolved to @Id[i64]
// so that trait.assume @Id[i64] inside @go matches the proven claim.

func.func nested @go(
  %arg0: !trait.poly<2>,
  %arg1: !trait.claim<@Prod[!trait.poly<2>]>,
  %arg2: !trait.claim<@Id[!trait.proj<@Prod[!trait.poly<2>], "Out">]>
) -> !trait.proj<@Prod[!trait.poly<2>], "Out"> {
  %0 = scf.execute_region -> !trait.proj<@Prod[!trait.poly<2>], "Out"> {
    %1 = trait.assume @Prod[!trait.poly<2>]
    %2 = trait.method.call %1 @Prod[!trait.poly<2>]::@get(%arg0)
      : (!trait.poly<2>) -> !trait.proj<@Prod[!trait.poly<2>], "Out">
    %3 = trait.assume @Id[!trait.proj<@Prod[!trait.poly<2>], "Out">]
    %4 = trait.method.call %3 @Id[!trait.proj<@Prod[!trait.poly<2>], "Out">]::@id(%2)
      : (!trait.proj<@Prod[!trait.poly<2>], "Out">) -> !trait.proj<@Prod[!trait.poly<2>], "Out">
    scf.yield %4 : !trait.proj<@Prod[!trait.poly<2>], "Out">
  }
  return %0 : !trait.proj<@Prod[!trait.poly<2>], "Out">
}

trait.proof @Id_blanket_Prod_i64_p proves @Id_blanket for @Id[i64] given []

// CHECK-LABEL: func.func nested @go
// CHECK-SAME: (%arg0: i64) -> i64
// CHECK: call @Prod_i64_get
// CHECK: call @Id_blanket_{{.*}}_id
// CHECK-NOT: trait.assume
// CHECK-NOT: trait.method.call

func.func nested @test_func_call() -> i64 {
  %0 = scf.execute_region -> i64 {
    %c42 = arith.constant 42 : i64
    %w1 = trait.witness @Prod_i64 for @Prod[i64]
    %w2 = trait.witness @Id_blanket_Prod_i64_p for @Id[i64]
    %r = trait.func.call @go(%c42, %w1, %w2)
      : (i64, !trait.claim<@Prod[i64] by @Prod_i64>, !trait.claim<@Id[i64] by @Id_blanket_Prod_i64_p>) -> i64
    scf.yield %r : i64
  }
  return %0 : i64
}

// CHECK-LABEL: func.func nested @test_func_call
// CHECK-NOT: trait.func.call

// --- Test 2: method.call with projection in where-clause claim ---
// A trait method whose where-clause requires @Id on a projection output.

!T = !trait.poly<6>
trait.trait @Pipeline[!T] {
  func.func nested @run(!T, !trait.claim<@Prod[!T]>, !trait.claim<@Id[!trait.proj<@Prod[!T], "Out">]>) -> !trait.proj<@Prod[!T], "Out">
}

trait.impl @Pipeline_i64 for @Pipeline[i64] {
  func.func nested @run(%x: i64, %prod: !trait.claim<@Prod[i64]>, %id: !trait.claim<@Id[i64]>) -> i64 {
    %produced = trait.method.call %prod @Prod[i64]::@get(%x) : (i64) -> i64
    %result = trait.method.call %id @Id[i64]::@id(%produced) : (i64) -> i64
    return %result : i64
  }
}

// CHECK-LABEL: func.func @test_method_call
// CHECK: call @Pipeline_i64_run
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.allege

func.func @test_method_call(%x: i64) -> i64 {
  %pipeline = trait.allege @Pipeline[i64]
  %prod = trait.allege @Prod[i64]
  %id = trait.allege @Id[i64]
  %r = trait.method.call %pipeline @Pipeline[i64]::@run(%x, %prod, %id)
    : (i64, !trait.claim<@Prod[i64]>, !trait.claim<@Id[i64]>) -> i64
  return %r : i64
}

// --- Test 3: chained projections ---
// Map's type argument is itself a projection: @Map[proj<@Prod[T], "Out">].
// Both Prod::Out and Map::Result must resolve for assumes to match.

func.func nested @chain(
  %x: !trait.poly<3>,
  %c1: !trait.claim<@Prod[!trait.poly<3>]>,
  %c2: !trait.claim<@Map[!trait.proj<@Prod[!trait.poly<3>], "Out">]>
) -> !trait.proj<@Map[!trait.proj<@Prod[!trait.poly<3>], "Out">], "Result"> {
  %1 = trait.assume @Prod[!trait.poly<3>]
  %2 = trait.method.call %1 @Prod[!trait.poly<3>]::@get(%x)
    : (!trait.poly<3>) -> !trait.proj<@Prod[!trait.poly<3>], "Out">
  %3 = trait.assume @Map[!trait.proj<@Prod[!trait.poly<3>], "Out">]
  %4 = trait.method.call %3 @Map[!trait.proj<@Prod[!trait.poly<3>], "Out">]::@apply(%2)
    : (!trait.proj<@Prod[!trait.poly<3>], "Out">) -> !trait.proj<@Map[!trait.proj<@Prod[!trait.poly<3>], "Out">], "Result">
  return %4 : !trait.proj<@Map[!trait.proj<@Prod[!trait.poly<3>], "Out">], "Result">
}

trait.proof @Map_i64_from_Prod proves @Map_i64 for @Map[i64] given []

// CHECK-LABEL: func.func nested @chain
// CHECK-SAME: (%arg0: i64) -> i64
// CHECK: call @Prod_i64_get
// CHECK: call @Map_i64_apply
// CHECK-NOT: trait.assume
// CHECK-NOT: trait.method.call

func.func nested @test_chain() -> i64 {
  %c7 = arith.constant 7 : i64
  %w1 = trait.witness @Prod_i64 for @Prod[i64]
  %w2 = trait.witness @Map_i64_from_Prod for @Map[i64]
  %r = trait.func.call @chain(%c7, %w1, %w2)
    : (i64, !trait.claim<@Prod[i64] by @Prod_i64>, !trait.claim<@Map[i64] by @Map_i64_from_Prod>) -> i64
  return %r : i64
}

// CHECK-LABEL: func.func nested @test_chain
// CHECK-NOT: trait.func.call
