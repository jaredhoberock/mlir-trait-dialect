// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!PartialEqS = !trait.poly<0>
!PartialEqO = !trait.poly<1>
// CHECK-NOT: trait.trait private @PartialEq
trait.trait private @PartialEq(%self_claim: !trait.claim<@PartialEq[!PartialEqS, !PartialEqO]>) {
  trait.method @eq(!PartialEqS, !PartialEqO) -> i1
  
  trait.method @ne(%self: !PartialEqS, %other: !PartialEqO) -> i1 {
    %equal = trait.method.call %self_claim @PartialEq[!PartialEqS,!PartialEqO]::@eq(%self, %other)
      : (!PartialEqS,!PartialEqO) -> i1

    %true = arith.constant 1 : i1
    %not_equal = arith.xori %equal, %true : i1
    trait.return %not_equal : i1
  }
}

// CHECK-NOT: trait.impl private @PartialEq
trait.impl private @PartialEq_impl_i32_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) {
  trait.method @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    trait.return %equal : i1
  }
}

!T = !trait.poly<0>

// CHECK-LABEL: func.func private @foo_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32_{{h[0-9a-f]+}}_eq
func.func private @foo(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %res = trait.method.call %c @PartialEq[!T,!T]::@eq(%x, %y)
    : (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @bar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @foo_{{.*}}
func.func @bar(%x: i32, %y: i32) -> i1 {
  %w = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  %res = trait.func.call @foo(%w, %x, %y)
    : (!trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>, i32, i32) -> i1

  return %res : i1
}

!EqS = !trait.poly<2>
// CHECK-NOT: @Eq
trait.trait private @Eq(%self: !trait.claim<@Eq[!trait.poly<0>]>) -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]> {
}

// CHECK-NOT: trait.impl private @Eq
trait.impl private @Eq_impl_i32(%self: !trait.claim<@Eq[i32]>) {
  %req0 = trait.allege @PartialEq[i32,i32]
  trait.return %req0 : !trait.claim<@PartialEq[i32,i32]>
}

// model Option<Ordering>
// 0: Less
// 1: Equal
// 2: Greater
// 3: None
!opt_ord = i2

!PartialOrdS = !trait.poly<3>
!PartialOrdO = !trait.poly<4>

// CHECK-NOT: trait.trait private @PartialOrd
trait.trait private @PartialOrd(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]> {
  trait.method @partial_cmp(!trait.poly<0>, !trait.poly<1>) -> !opt_ord

  trait.method @lt(%self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
    %cmp = trait.method.call %self_claim @PartialOrd[!trait.poly<0>,!trait.poly<1>]::@partial_cmp(%self, %other)
      : (!trait.poly<0>,!trait.poly<1>) -> !opt_ord

    %ord_lt = arith.constant 0 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_lt : !opt_ord
    trait.return %res : i1
  }

  trait.method @le(%self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
    %partial_eq = trait.project %self_claim[0]
      : !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>
      -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>

    %lt = trait.method.call %self_claim @PartialOrd[!trait.poly<0>,!trait.poly<1>]::@lt(%self, %other)
      : (!trait.poly<0>,!trait.poly<1>) -> i1

    %eq = trait.method.call %partial_eq @PartialEq[!trait.poly<0>,!trait.poly<1>]::@eq(%self, %other)
      : (!trait.poly<0>,!trait.poly<1>) -> i1

    %res = arith.ori %lt, %eq : i1
    trait.return %res : i1
  }

  trait.method @gt(%self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
    %cmp = trait.method.call %self_claim @PartialOrd[!trait.poly<0>,!trait.poly<1>]::@partial_cmp(%self, %other)
      : (!trait.poly<0>,!trait.poly<1>) -> !opt_ord

    %ord_gt = arith.constant 2 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_gt : !opt_ord
    trait.return %res : i1
  }

  trait.method @ge(%self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
    %partial_eq = trait.project %self_claim[0]
      : !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>
      -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>

    %gt = trait.method.call %self_claim @PartialOrd[!trait.poly<0>,!trait.poly<1>]::@gt(%self, %other)
      : (!trait.poly<0>,!trait.poly<1>) -> i1

    %eq = trait.method.call %partial_eq @PartialEq[!trait.poly<0>,!trait.poly<1>]::@eq(%self, %other)
      : (!trait.poly<0>,!trait.poly<1>) -> i1

    %res = arith.ori %gt, %eq : i1
    trait.return %res : i1
  }
}

// CHECK-NOT: trait.impl private @PartialOrd
trait.impl private @PartialOrd_impl_i32_i32(%self: !trait.claim<@PartialOrd[i32, i32]>) {
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
  %req0 = trait.allege @PartialEq[i32,i32]
  trait.return %req0 : !trait.claim<@PartialEq[i32,i32]>
}

// model Ordering
// 0: Less
// 1: Equal
// 2: Greater
!ord = i2

!OrdS = !trait.poly<5>
// CHECK-NOT: trait.trait private @Ord
trait.trait private @Ord(%self_claim: !trait.claim<@Ord[!trait.poly<0>]>) -> (!trait.claim<@Eq[!trait.poly<0>]>, !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>) {
  trait.method @cmp(!trait.poly<0>, !trait.poly<0>) -> !ord

  trait.method @max(%self: !trait.poly<0>, %other: !trait.poly<0>) -> !trait.poly<0> {
    %partial_ord = trait.project %self_claim[1]
      : !trait.claim<@Ord[!trait.poly<0>]>
      -> !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>

    %cond = trait.method.call %partial_ord @PartialOrd[!trait.poly<0>,!trait.poly<0>]::@gt(%self, %other)
      : (!trait.poly<0>,!trait.poly<0>) -> i1

    %res = scf.if %cond -> !trait.poly<0> {
      scf.yield %self : !trait.poly<0>
    } else {
      scf.yield %other : !trait.poly<0>
    }

    trait.return %res : !trait.poly<0>
  }

  trait.method @min(%self: !trait.poly<0>, %other: !trait.poly<0>) -> !trait.poly<0> {
    %partial_ord = trait.project %self_claim[1]
      : !trait.claim<@Ord[!trait.poly<0>]>
      -> !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>

    %cond = trait.method.call %partial_ord @PartialOrd[!trait.poly<0>,!trait.poly<0>]::@le(%self, %other)
      : (!trait.poly<0>,!trait.poly<0>) -> i1

    %res = scf.if %cond -> !trait.poly<0> {
      scf.yield %self: !trait.poly<0>
    } else {
      scf.yield %other: !trait.poly<0>
    }

    trait.return %res : !trait.poly<0>
  }
}

// CHECK-NOT: trait.impl private @Ord
trait.impl private @Ord_impl_i32(%self: !trait.claim<@Ord[i32]>) {
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
  %req0 = trait.allege @Eq[i32]
  %req1 = trait.allege @PartialOrd[i32,i32]
  trait.return %req0, %req1 : !trait.claim<@Eq[i32]>, !trait.claim<@PartialOrd[i32,i32]>
}

// CHECK-LABEL: func.func @max
// CHECK-NOT: trait.claim
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord_{{h[0-9a-f]+}}_max
func.func @max(%a: i32, %b: i32) -> i32 {
  %ord = trait.allege @Ord[i32]
  %res = trait.method.call %ord @Ord[i32]::@max(%a, %b)
    : (i32,i32) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @min
// CHECK-NOT: trait.claim
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord_{{h[0-9a-f]+}}_min
func.func @min(%a: i32, %b: i32) -> i32 {
  %ord = trait.allege @Ord[i32]
  %res = trait.method.call %ord @Ord[i32]::@min(%a, %b)
    : (i32,i32) -> i32
  return %res : i32
}
