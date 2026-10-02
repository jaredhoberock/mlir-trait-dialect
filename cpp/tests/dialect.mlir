// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: test everything

// CHECK-LABEL: trait private @PartialEq(%self: !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>)
!PartialEqS = !trait.poly<0>
!PartialEqO = !trait.poly<1>
trait.trait private @PartialEq(%self_claim: !trait.claim<@PartialEq[!PartialEqS, !PartialEqO]>) {
  // CHECK-LABEL: trait.method @eq
  trait.method @eq(!PartialEqS, !PartialEqO) -> i1
  
  // CHECK-LABEL: trait.method @ne
  trait.method @ne(%self: !PartialEqS, %other: !PartialEqO) -> i1 {
    %equal = trait.method.call %self_claim @PartialEq[!PartialEqS,!PartialEqO]::@eq(%self, %other)
      : (!PartialEqS,!PartialEqO) -> i1
    %true = arith.constant 1 : i1
    %not_equal = arith.xori %equal, %true : i1
    trait.return %not_equal : i1
  }
}

// CHECK-LABEL: trait.impl private @PartialEq_impl_i32_i32(%self: !trait.claim<@PartialEq[i32, i32]>
trait.impl private @PartialEq_impl_i32_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) {
  // CHECK-LABEL: trait.method @eq
  trait.method @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    trait.return %equal : i1
  }
}

// CHECK-LABEL: func.func @foo
!T = !trait.poly<0>
!W = !trait.claim<@PartialEq[!T,!T]>
func.func @foo(%w: !W, %x: !T, %y: !T) -> i1 {
  // CHECK: %[[RES:.*]] = trait.method.call %{{.*}} @PartialEq
  %res = trait.method.call %w @PartialEq[!T,!T]::@eq(%x, %y)
    : (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @bar
func.func @bar(%x: i32, %y: i32) -> i1 {
  %w = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]

  // CHECK: %[[RES:.*]] = trait.func.call @foo
  %res = trait.func.call @foo(%w, %x, %y)
    : (!trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>, i32, i32) -> i1

  return %res : i1
}

// CHECK-LABEL: trait private @Eq(%self: !trait.claim<@Eq[!trait.poly<2>]>) -> !trait.claim<@PartialEq[!trait.poly<2>, !trait.poly<2>]>
!EqS = !trait.poly<2>
trait.trait private @Eq(%self: !trait.claim<@Eq[!EqS]>) -> !trait.claim<@PartialEq[!EqS,!EqS]> {
}

// CHECK-LABEL: impl private @Eq_impl_i32(%self: !trait.claim<@Eq[i32]>)
trait.impl private @Eq_impl_i32(%self: !trait.claim<@Eq[i32]>) {
  %partial_eq = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  trait.return %partial_eq : !trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>
}

// model Option<Ordering>
// 0: Less
// 1: Equal
// 2: Greater
// 3: None
!opt_ord = i2

// CHECK-LABEL: trait private @PartialOrd(%self: !trait.claim<@PartialOrd[!trait.poly<3>, !trait.poly<4>]>) -> !trait.claim<@PartialEq
!PartialOrdS = !trait.poly<3>
!PartialOrdO = !trait.poly<4>
trait.trait private @PartialOrd(%self_claim: !trait.claim<@PartialOrd[!PartialOrdS, !PartialOrdO]>) -> !trait.claim<@PartialEq[!PartialOrdS,!PartialOrdO]> {
  // CHECK-LABEL: trait.method @partial_cmp
  trait.method @partial_cmp(!PartialOrdS, !PartialOrdO) -> !opt_ord

  // CHECK-LABEL: trait.method @lt
  trait.method @lt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %cmp = trait.method.call %self_claim @PartialOrd[!PartialOrdS,!PartialOrdO]::@partial_cmp(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_lt = arith.constant 0 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_lt : !opt_ord
    trait.return %res : i1
  }

  // CHECK-LABEL: trait.method @le
  trait.method @le(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %partial_eq_p = trait.project %self_claim[0]
      : !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]>
      -> !trait.claim<@PartialEq[!PartialOrdS,!PartialOrdO]>

    %lt = trait.method.call %self_claim @PartialOrd[!PartialOrdS,!PartialOrdO]::@lt(%self, %other)
      : (!PartialOrdS, !PartialOrdO) -> i1

    %eq = trait.method.call %partial_eq_p @PartialEq[!PartialOrdS,!PartialOrdO]::@eq(%self, %other)
      : (!PartialOrdS, !PartialOrdO) -> i1

    %res = arith.ori %lt, %eq : i1
    trait.return %res : i1
  }

  // CHECK-LABEL: trait.method @gt
  trait.method @gt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %cmp = trait.method.call %self_claim @PartialOrd[!PartialOrdS,!PartialOrdO]::@partial_cmp(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_gt = arith.constant 2 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_gt : !opt_ord
    trait.return %res : i1
  }

  // CHECK-LABEL: trait.method @ge
  trait.method @ge(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %partial_eq = trait.project %self_claim[0]
      : !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]>
      -> !trait.claim<@PartialEq[!PartialOrdS,!PartialOrdO]>

    %gt = trait.method.call %self_claim @PartialOrd[!PartialOrdS,!PartialOrdO]::@gt(%self, %other)
      : (!PartialOrdS, !PartialOrdO) -> i1

    %eq = trait.method.call %partial_eq @PartialEq[!PartialOrdS,!PartialOrdO]::@eq(%self, %other)
      : (!PartialOrdS, !PartialOrdO) -> i1

    %res = arith.ori %gt, %eq : i1
    trait.return %res : i1
  }
}

// CHECK-LABEL: trait.impl private @PartialOrd_impl_i32_i32(%self: !trait.claim<@PartialOrd[i32, i32]>
trait.impl private @PartialOrd_impl_i32_i32(%self: !trait.claim<@PartialOrd[i32, i32]>) {
  // CHECK-LABEL: trait.method @partial_cmp
  trait.method @partial_cmp(%a: i32, %b: i32) -> !opt_ord {
    %c_lt = arith.constant 0 : !opt_ord
    %c_eq = arith.constant 1 : !opt_ord
    %c_gt = arith.constant 2 : !opt_ord

    %lt = arith.cmpi slt, %a, %b : i32
    %eq = arith.cmpi eq,  %a, %b : i32
    %gt_or_lt = arith.select %lt, %c_lt, %c_gt : !opt_ord
    %res = arith.select %eq, %c_eq, %gt_or_lt : !opt_ord
    trait.return %res : !opt_ord
  }
  %partial_eq = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  trait.return %partial_eq : !trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>
}

// model Ordering
// 0: Less
// 1: Equal
// 2: Greater
!ord = i2

// CHECK-LABEL: trait private @Ord(%self: !trait.claim<@Ord[!trait.poly<5>]>) -> (!trait.claim<@Eq[!trait.poly<5>]>, !trait.claim<@PartialOrd[!trait.poly<5>, !trait.poly<5>]>)
!OrdS = !trait.poly<5>
trait.trait private @Ord(%self_claim: !trait.claim<@Ord[!OrdS]>) -> (!trait.claim<@Eq[!OrdS]>, !trait.claim<@PartialOrd[!OrdS,!OrdS]>) {
  // CHECK-LABEL: trait.method @cmp
  trait.method @cmp(!OrdS, !OrdS) -> !ord

  // CHECK-LABEL: trait.method @max
  trait.method @max(%self: !OrdS, %other: !OrdS) -> !OrdS {
    %partial_ord_p = trait.project %self_claim[1]
      : !trait.claim<@Ord[!OrdS]>
      -> !trait.claim<@PartialOrd[!OrdS,!OrdS]>

    %cond = trait.method.call %partial_ord_p @PartialOrd[!OrdS,!OrdS]::@gt(%self, %other)
      : (!OrdS,!OrdS) -> i1

    %res = scf.if %cond -> !OrdS {
      scf.yield %self : !OrdS
    } else {
      scf.yield %other : !OrdS
    }

    trait.return %res : !OrdS
  }

  // CHECK-LABEL: trait.method @min
  trait.method @min(%self: !OrdS, %other: !OrdS) -> !OrdS {
    %partial_ord = trait.project %self_claim[1]
      : !trait.claim<@Ord[!OrdS]>
      -> !trait.claim<@PartialOrd[!OrdS,!OrdS]>

    %cond = trait.method.call %partial_ord @PartialOrd[!OrdS,!OrdS]::@le(%self, %other)
      : (!OrdS,!OrdS) -> i1

    %res = scf.if %cond -> !OrdS {
      scf.yield %self: !OrdS
    } else {
      scf.yield %other: !OrdS
    }

    trait.return %res : !OrdS
  }
}

// CHECK-LABEL: trait.impl private @Ord_impl_i32(%self: !trait.claim<@Ord[i32]>
trait.impl private @Ord_impl_i32(%self: !trait.claim<@Ord[i32]>) {
  // CHECK-LABEL: trait.method @cmp
  trait.method @cmp(%a: i32, %b: i32) -> !ord {
    %lt = arith.cmpi slt, %a, %b : i32
    %eq = arith.cmpi eq,  %a, %b : i32

    %c_lt = arith.constant 0 : !ord
    %c_eq = arith.constant 1 : !ord
    %c_gt = arith.constant 2 : !ord

    %gt_or_lt = arith.select %lt, %c_lt, %c_gt : !ord
    %res = arith.select %eq, %c_eq, %gt_or_lt : !ord
    trait.return %res : !ord
  }
  %eq_p = trait.witness @Eq_impl_i32_p for @Eq[i32]
  %partial_ord_p = trait.witness @PartialOrd_impl_i32_i32_p for @PartialOrd[i32,i32]
  trait.return %eq_p, %partial_ord_p : !trait.claim<@Eq[i32] by @Eq_impl_i32_p>, !trait.claim<@PartialOrd[i32,i32] by @PartialOrd_impl_i32_i32_p>
}

// CHECK-LABEL: trait.proof private @PartialOrd_impl_i32_i32_p
trait.proof private @PartialOrd_impl_i32_i32_p {
  %d = trait.derive @PartialOrd[i32, i32] from @PartialOrd_impl_i32_i32 given()
  trait.return %d : !trait.claim<@PartialOrd[i32, i32]>
}

// CHECK-LABEL: trait.proof private @Eq_impl_i32_p
trait.proof private @Eq_impl_i32_p {
  %d = trait.derive @Eq[i32] from @Eq_impl_i32 given()
  trait.return %d : !trait.claim<@Eq[i32]>
}

// CHECK-LABEL: trait.proof private @Ord_impl_i32_p
trait.proof private @Ord_impl_i32_p {
  %d = trait.derive @Ord[i32] from @Ord_impl_i32 given()
  trait.return %d : !trait.claim<@Ord[i32]>
}

// CHECK-LABEL: func.func @max
func.func @max(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.method.call %ord_p @Ord[i32]::@max(%a, %b)
    : (i32, i32) -> i32
    by @Ord_impl_i32_p

  return %res : i32
}

// CHECK-LABEL: func.func @min
func.func @min(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.method.call %ord_p @Ord[i32]::@min(%a, %b)
    : (i32, i32) -> i32
    by @Ord_impl_i32_p

  return %res : i32
}
