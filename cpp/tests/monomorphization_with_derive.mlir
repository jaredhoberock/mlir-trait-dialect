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

trait.trait @Trait [!T0] {
  func.func private @method(!T0) -> i32
}

// Unconditional base impl for i32
trait.impl @Trait_impl_i32 for @Trait[i32] {
  func.func @method(%self: i32) -> i32 {
    %res = arith.constant 42 : i32
    return %res : i32
  }
}

// Conditional impl: Trait[tuple<U>] given Trait[U]
!T1 = !trait.poly<1>
trait.impl @Trait_impl_tuple for @Trait[tuple<!T1>] where [@Trait[!T1]] {
  func.func @method(%self: tuple<!T1>) -> i32 {
    %a = trait.assume @Trait[!T1]
    %res = arith.constant 1 : i32
    return %res : i32
  }
}

//===----------------------------------------------------------------------===//
// 1. Basic derive: Trait[!T] -> Trait[tuple<!T>]
//===----------------------------------------------------------------------===//

!T2 = !trait.poly<2>

func.func @poly_fn(%arg: tuple<!T2>, %t_claim: !trait.claim<@Trait[!T2]>) -> i32 {
  %d = trait.derive @Trait[tuple<!T2>] from @Trait_impl_tuple given(%t_claim) : (!trait.claim<@Trait[!T2]>)
  %res = trait.method.call %d @Trait[tuple<!T2>]::@method(%arg)
    : (tuple<!T2>) -> i32
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

func.func @double_wrap(%arg: tuple<tuple<!T3>>, %t_claim: !trait.claim<@Trait[!T3]>) -> i32 {
  %d1 = trait.derive @Trait[tuple<!T3>] from @Trait_impl_tuple given(%t_claim) : (!trait.claim<@Trait[!T3]>)
  %d2 = trait.derive @Trait[tuple<tuple<!T3>>] from @Trait_impl_tuple given(%d1) : (!trait.claim<@Trait[tuple<!T3>]>)
  %res = trait.method.call %d2 @Trait[tuple<tuple<!T3>>]::@method(%arg)
    : (tuple<tuple<!T3>>) -> i32
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

trait.trait @TraitA [!T4] {
  func.func private @method_a(!T4) -> i32
}

!T5 = !trait.poly<5>

trait.trait @TraitB [!T5] {
  func.func private @method_b(!T5) -> i32
}

trait.impl @TraitA_impl_i32 for @TraitA[i32] {
  func.func @method_a(%self: i32) -> i32 {
    %res = arith.constant 10 : i32
    return %res : i32
  }
}

// TraitB[U] holds whenever TraitA[U] holds
!T6 = !trait.poly<6>
trait.impl @TraitB_from_TraitA for @TraitB[!T6] where [@TraitA[!T6]] {
  func.func @method_b(%self: !T6) -> i32 {
    %a = trait.assume @TraitA[!T6]
    %res = trait.method.call %a @TraitA[!T6]::@method_a(%self)
      : (!T6) -> i32
    return %res : i32
  }
}

!T7 = !trait.poly<7>

func.func @cross_trait(%arg: !T7, %a_claim: !trait.claim<@TraitA[!T7]>) -> i32 {
  %b = trait.derive @TraitB[!T7] from @TraitB_from_TraitA given(%a_claim) : (!trait.claim<@TraitA[!T7]>)
  %res = trait.method.call %b @TraitB[!T7]::@method_b(%arg)
    : (!T7) -> i32
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

trait.trait @TraitC [!T8] {
  func.func private @method_c(!T8) -> i32
}

trait.impl @TraitC_impl_i32 for @TraitC[i32] {
  func.func @method_c(%self: i32) -> i32 {
    %res = arith.constant 20 : i32
    return %res : i32
  }
}

// TraitC[tuple<U>] holds whenever both TraitA[U] and TraitC[U] hold
!T9 = !trait.poly<9>
trait.impl @TraitC_impl_tuple for @TraitC[tuple<!T9>] where [@TraitA[!T9], @TraitC[!T9]] {
  func.func @method_c(%self: tuple<!T9>) -> i32 {
    %a = trait.assume @TraitA[!T9]
    %c = trait.assume @TraitC[!T9]
    %res = arith.constant 30 : i32
    return %res : i32
  }
}

!T10 = !trait.poly<10>

func.func @multi_assumption(%arg: tuple<!T10>,
                             %a_claim: !trait.claim<@TraitA[!T10]>,
                             %c_claim: !trait.claim<@TraitC[!T10]>) -> i32 {
  %d = trait.derive @TraitC[tuple<!T10>] from @TraitC_impl_tuple given(%a_claim, %c_claim)
    : (!trait.claim<@TraitA[!T10]>, !trait.claim<@TraitC[!T10]>)
  %res = trait.method.call %d @TraitC[tuple<!T10>]::@method_c(%arg)
    : (tuple<!T10>) -> i32
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
