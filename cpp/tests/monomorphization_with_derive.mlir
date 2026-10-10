// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Verifies that trait.derive is lowered through monomorphization:
// the derive op should be replaced by a trait.witness backed by a minted trait.proof,
// then erased along with all other trait infrastructure.
//
// Tests cover:
//   1. Basic derive (single conditional impl, one assumption)
//   2. Chained derives (two successive derives building nested types)
//   3. Cross-trait derive (derive TraitB from a TraitA claim)
//   4. Multiple assumptions (derive from an impl with two where clauses)

// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.derive
// CHECK-NOT: trait.witness
// CHECK-NOT: trait.proof
// CHECK-NOT: trait.allege
// CHECK-NOT: trait.assume

//===----------------------------------------------------------------------===//
// Shared trait and impl definitions
//===----------------------------------------------------------------------===//

!T0 = !trait.poly<0>

trait.trait private @Trait(%self: !trait.claim<@Trait[!T0]>) {
  trait.method @method(!T0) -> i32
}

// Unconditional base impl for i32
trait.impl private @Trait_impl_i32(%self_claim: !trait.claim<@Trait[i32]>) {
  trait.method @method(%self: i32) -> i32 {
    %res = arith.constant 42 : i32
    trait.return %res : i32
  }
}

// Conditional impl: Trait[tuple<U>] given Trait[U]
!T1 = !trait.poly<1>
trait.impl private @Trait_impl_tuple(%self_claim: !trait.claim<@Trait[tuple<!trait.poly<0>>]>, %trait: !trait.claim<@Trait[!trait.poly<0>]>) {
  trait.method @method(%self: tuple<!trait.poly<0>>) -> i32 {
    %res = arith.constant 1 : i32
    trait.return %res : i32
  }
}

//===----------------------------------------------------------------------===//
// 1. Basic derive: Trait[!T] -> Trait[tuple<!T>]
//===----------------------------------------------------------------------===//

!T2 = !trait.poly<2>

func.func private @poly_fn(%arg: tuple<!trait.poly<0>>, %t_claim: !trait.claim<@Trait[!trait.poly<0>]>) -> i32 {
  %d = trait.derive @Trait[tuple<!trait.poly<0>>] from @Trait_impl_tuple[!trait.poly<0>] given(%t_claim) : (!trait.claim<@Trait[!trait.poly<0>]>)
  %res = trait.method.call %d @Trait[tuple<!trait.poly<0>>]::@method(%arg)
    : (tuple<!trait.poly<0>>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @test_basic_derive
// CHECK: call @poly_fn
func.func @test_basic_derive(%arg: tuple<i32>) -> i32 {
  %a = trait.allege @Trait[i32]
  %res = trait.func.call @poly_fn(%arg, %a)
    : (tuple<i32>, !trait.claim<@Trait[i32]>) -> i32
  return %res : i32
}

//===----------------------------------------------------------------------===//
// 2. Chained derives: Trait[!T] -> Trait[tuple<!T>] -> Trait[tuple<tuple<!T>>]
//===----------------------------------------------------------------------===//

!T3 = !trait.poly<3>

func.func private @double_wrap(%arg: tuple<tuple<!trait.poly<0>>>, %t_claim: !trait.claim<@Trait[!trait.poly<0>]>) -> i32 {
  %d1 = trait.derive @Trait[tuple<!trait.poly<0>>] from @Trait_impl_tuple[!trait.poly<0>] given(%t_claim) : (!trait.claim<@Trait[!trait.poly<0>]>)
  %d2 = trait.derive @Trait[tuple<tuple<!trait.poly<0>>>] from @Trait_impl_tuple[tuple<!trait.poly<0>>] given(%d1) : (!trait.claim<@Trait[tuple<!trait.poly<0>>]>)
  %res = trait.method.call %d2 @Trait[tuple<tuple<!trait.poly<0>>>]::@method(%arg)
    : (tuple<tuple<!trait.poly<0>>>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @test_chained_derive
// CHECK: call @double_wrap
func.func @test_chained_derive(%arg: tuple<tuple<i32>>) -> i32 {
  %a = trait.allege @Trait[i32]
  %res = trait.func.call @double_wrap(%arg, %a)
    : (tuple<tuple<i32>>, !trait.claim<@Trait[i32]>) -> i32
  return %res : i32
}

//===----------------------------------------------------------------------===//
// 3. Cross-trait derive: TraitA[!T] claim used to derive TraitB[!T]
//===----------------------------------------------------------------------===//

!T4 = !trait.poly<4>

trait.trait private @TraitA(%self: !trait.claim<@TraitA[!trait.poly<0>]>) {
  trait.method @method_a(!trait.poly<0>) -> i32
}

!T5 = !trait.poly<5>

trait.trait private @TraitB(%self: !trait.claim<@TraitB[!trait.poly<0>]>) {
  trait.method @method_b(!trait.poly<0>) -> i32
}

trait.impl private @TraitA_impl_i32(%self_claim: !trait.claim<@TraitA[i32]>) {
  trait.method @method_a(%self: i32) -> i32 {
    %res = arith.constant 10 : i32
    trait.return %res : i32
  }
}

// TraitB[U] holds whenever TraitA[U] holds
!T6 = !trait.poly<6>
trait.impl private @TraitB_from_TraitA(%self_claim: !trait.claim<@TraitB[!trait.poly<0>]>, %traita: !trait.claim<@TraitA[!trait.poly<0>]>) {
  trait.method @method_b(%self: !trait.poly<0>) -> i32 {
    %res = trait.method.call %traita @TraitA[!trait.poly<0>]::@method_a(%self)
      : (!trait.poly<0>) -> i32
    trait.return %res : i32
  }
}

!T7 = !trait.poly<7>

func.func private @cross_trait(%arg: !trait.poly<0>, %a_claim: !trait.claim<@TraitA[!trait.poly<0>]>) -> i32 {
  %b = trait.derive @TraitB[!trait.poly<0>] from @TraitB_from_TraitA[!trait.poly<0>] given(%a_claim) : (!trait.claim<@TraitA[!trait.poly<0>]>)
  %res = trait.method.call %b @TraitB[!trait.poly<0>]::@method_b(%arg)
    : (!trait.poly<0>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @test_cross_trait_derive
// CHECK: call @cross_trait
func.func @test_cross_trait_derive(%arg: i32) -> i32 {
  %a = trait.allege @TraitA[i32]
  %res = trait.func.call @cross_trait(%arg, %a)
    : (i32, !trait.claim<@TraitA[i32]>) -> i32
  return %res : i32
}

//===----------------------------------------------------------------------===//
// 4. Multiple assumptions: derive from an impl with two where clauses
//===----------------------------------------------------------------------===//

!T8 = !trait.poly<8>

trait.trait private @TraitC(%self: !trait.claim<@TraitC[!trait.poly<0>]>) {
  trait.method @method_c(!trait.poly<0>) -> i32
}

trait.impl private @TraitC_impl_i32(%self_claim: !trait.claim<@TraitC[i32]>) {
  trait.method @method_c(%self: i32) -> i32 {
    %res = arith.constant 20 : i32
    trait.return %res : i32
  }
}

// TraitC[tuple<U>] holds whenever both TraitA[U] and TraitC[U] hold
!T9 = !trait.poly<9>
trait.impl private @TraitC_impl_tuple(%self_claim: !trait.claim<@TraitC[tuple<!trait.poly<0>>]>, %traita: !trait.claim<@TraitA[!trait.poly<0>]>, %traitc: !trait.claim<@TraitC[!trait.poly<0>]>) {
  trait.method @method_c(%self: tuple<!trait.poly<0>>) -> i32 {
    %res = arith.constant 30 : i32
    trait.return %res : i32
  }
}

!T10 = !trait.poly<10>

func.func private @multi_assumption(%arg: tuple<!trait.poly<0>>,
                             %a_claim: !trait.claim<@TraitA[!trait.poly<0>]>,
                             %c_claim: !trait.claim<@TraitC[!trait.poly<0>]>) -> i32 {
  %d = trait.derive @TraitC[tuple<!trait.poly<0>>] from @TraitC_impl_tuple[!trait.poly<0>] given(%a_claim, %c_claim)
    : (!trait.claim<@TraitA[!trait.poly<0>]>, !trait.claim<@TraitC[!trait.poly<0>]>)
  %res = trait.method.call %d @TraitC[tuple<!trait.poly<0>>]::@method_c(%arg)
    : (tuple<!trait.poly<0>>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @test_multi_assumption_derive
// CHECK: call @multi_assumption
func.func @test_multi_assumption_derive(%arg: tuple<i32>) -> i32 {
  %a = trait.allege @TraitA[i32]
  %c = trait.allege @TraitC[i32]
  %res = trait.func.call @multi_assumption(%arg, %a, %c)
    : (tuple<i32>, !trait.claim<@TraitA[i32]>, !trait.claim<@TraitC[i32]>) -> i32
  return %res : i32
}
