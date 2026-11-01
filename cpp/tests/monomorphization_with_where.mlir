// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!PartialEqS = !trait.poly<0>
!PartialEqO = !trait.poly<1>
// CHECK-NOT: trait.trait private @PartialEq
trait.trait private @PartialEq(!trait.claim<@PartialEq[!PartialEqS, !PartialEqO]>) members [@PartialEq.eq, default @PartialEq.ne]
func.func private @PartialEq.eq(!trait.claim<@PartialEq[!PartialEqS, !PartialEqO]>, !PartialEqS, !PartialEqO) -> i1
func.func private @PartialEq.ne(%self_claim: !trait.claim<@PartialEq[!PartialEqS, !PartialEqO]>, %self: !PartialEqS, %other: !PartialEqO) -> i1 {
  %equal = trait.func.call @PartialEq.eq(%self_claim, %self, %other) : (!trait.claim<@PartialEq[!PartialEqS,!PartialEqO]>, !PartialEqS, !PartialEqO) -> i1
  %true = arith.constant 1 : i1
  %not_equal = arith.xori %equal, %true : i1
  return %not_equal : i1
}

// CHECK-NOT: trait.impl private @PartialEq
trait.impl private @PartialEq_impl_i32_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) members [@PartialEq.eq -> @PartialEq_impl_i32_i32.eq] {
}
func.func private @PartialEq_impl_i32_i32.eq(%self_claim: !trait.claim<@PartialEq[i32, i32]>, %self: i32, %other: i32) -> i1 {
  %equal = arith.cmpi eq, %self, %other : i32
  return %equal : i1
}

!T = !trait.poly<0>

// CHECK-LABEL: func.func private @foo_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32.eq_{{h[0-9a-f]+}}
func.func private @foo(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %res = trait.func.call @PartialEq.eq(%c, %x, %y) : (!trait.claim<@PartialEq[!T,!T]>, !T, !T) -> i1
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
trait.trait private @Eq(!trait.claim<@Eq[!trait.poly<0>]>) -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>

// CHECK-NOT: trait.impl private @Eq
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

!PartialOrdS = !trait.poly<3>
!PartialOrdO = !trait.poly<4>

// CHECK-NOT: trait.trait private @PartialOrd
trait.trait private @PartialOrd(!trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]> members [@PartialOrd.partial_cmp, default @PartialOrd.lt, default @PartialOrd.le, default @PartialOrd.gt, default @PartialOrd.ge]
func.func private @PartialOrd.partial_cmp(!trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> !opt_ord
func.func private @PartialOrd.lt(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, %self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
  %cmp = trait.func.call @PartialOrd.partial_cmp(%self_claim, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>,!trait.poly<1>) -> !opt_ord

  %ord_lt = arith.constant 0 : !opt_ord
  %res = arith.cmpi eq, %cmp, %ord_lt : !opt_ord
  return %res : i1
}
func.func private @PartialOrd.le(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, %self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
  %partial_eq = trait.project %self_claim[0]
    : !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>
    -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>

  %lt = trait.func.call @PartialOrd.lt(%self_claim, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>,!trait.poly<1>) -> i1

  %eq = trait.func.call @PartialEq.eq(%partial_eq, %self, %other) : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>,!trait.poly<1>) -> i1

  %res = arith.ori %lt, %eq : i1
  return %res : i1
}
func.func private @PartialOrd.gt(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, %self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
  %cmp = trait.func.call @PartialOrd.partial_cmp(%self_claim, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>,!trait.poly<1>) -> !opt_ord

  %ord_gt = arith.constant 2 : !opt_ord
  %res = arith.cmpi eq, %cmp, %ord_gt : !opt_ord
  return %res : i1
}
func.func private @PartialOrd.ge(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, %self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
  %partial_eq = trait.project %self_claim[0]
    : !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>
    -> !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>

  %gt = trait.func.call @PartialOrd.gt(%self_claim, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>,!trait.poly<1>) -> i1

  %eq = trait.func.call @PartialEq.eq(%partial_eq, %self, %other) : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>,!trait.poly<1>) -> i1

  %res = arith.ori %gt, %eq : i1
  return %res : i1
}

// CHECK-NOT: trait.impl private @PartialOrd
trait.impl private @PartialOrd_impl_i32_i32(%self: !trait.claim<@PartialOrd[i32, i32]>) members [@PartialOrd.partial_cmp -> @PartialOrd_impl_i32_i32.partial_cmp] {
  %partial_eq = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  trait.return %partial_eq : !trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>
}
func.func private @PartialOrd_impl_i32_i32.partial_cmp(%self: !trait.claim<@PartialOrd[i32, i32]>, %a: i32, %b: i32) -> !opt_ord {
  %c_lt = arith.constant 0 : !opt_ord
  %c_eq = arith.constant 1 : !opt_ord
  %c_gt = arith.constant 2 : !opt_ord

  %lt = arith.cmpi slt, %a, %b : i32
  %eq = arith.cmpi eq,  %a, %b : i32
  %gt_or_lt = arith.select %lt, %c_lt, %c_gt : !opt_ord
  %res = arith.select %eq, %c_eq, %gt_or_lt : !opt_ord
  return %res : !opt_ord
}

// model Ordering
// 0: Less
// 1: Equal
// 2: Greater
!ord = i2

!OrdS = !trait.poly<5>
// CHECK-NOT: trait.trait private @Ord
trait.trait private @Ord(!trait.claim<@Ord[!trait.poly<0>]>) -> (!trait.claim<@Eq[!trait.poly<0>]>, !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>) members [@Ord.cmp, default @Ord.max, default @Ord.min]
func.func private @Ord.cmp(!trait.claim<@Ord[!trait.poly<0>]>, !trait.poly<0>, !trait.poly<0>) -> !ord
func.func private @Ord.max(%self_claim: !trait.claim<@Ord[!trait.poly<0>]>, %self: !trait.poly<0>, %other: !trait.poly<0>) -> !trait.poly<0> {
  %partial_ord = trait.project %self_claim[1]
    : !trait.claim<@Ord[!trait.poly<0>]>
    -> !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>

  %cond = trait.func.call @PartialOrd.gt(%partial_ord, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>,!trait.poly<0>) -> i1

  %res = scf.if %cond -> !trait.poly<0> {
    scf.yield %self : !trait.poly<0>
  } else {
    scf.yield %other : !trait.poly<0>
  }

  return %res : !trait.poly<0>
}
func.func private @Ord.min(%self_claim: !trait.claim<@Ord[!trait.poly<0>]>, %self: !trait.poly<0>, %other: !trait.poly<0>) -> !trait.poly<0> {
  %partial_ord = trait.project %self_claim[1]
    : !trait.claim<@Ord[!trait.poly<0>]>
    -> !trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>

  %cond = trait.func.call @PartialOrd.le(%partial_ord, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>,!trait.poly<0>) -> i1

  %res = scf.if %cond -> !trait.poly<0> {
    scf.yield %self: !trait.poly<0>
  } else {
    scf.yield %other: !trait.poly<0>
  }

  return %res : !trait.poly<0>
}

// CHECK-NOT: trait.impl private @Ord
trait.impl private @Ord_impl_i32(%self: !trait.claim<@Ord[i32]>) members [@Ord.cmp -> @Ord_impl_i32.cmp] {
  %eq = trait.witness @Eq_impl_i32_p for @Eq[i32]
  %partial_ord = trait.witness @PartialOrd_impl_i32_i32_p for @PartialOrd[i32,i32]
  trait.return %eq, %partial_ord : !trait.claim<@Eq[i32] by @Eq_impl_i32_p>, !trait.claim<@PartialOrd[i32,i32] by @PartialOrd_impl_i32_i32_p>
}
func.func private @Ord_impl_i32.cmp(%self: !trait.claim<@Ord[i32]>, %a: i32, %b: i32) -> !ord {
  %lt = arith.cmpi slt, %a, %b : i32
  %eq = arith.cmpi eq,  %a, %b : i32

  %c_lt = arith.constant 0 : !ord
  %c_eq = arith.constant 1 : !ord
  %c_gt = arith.constant 2 : !ord

  %gt_or_lt = arith.select %lt, %c_lt, %c_gt : !ord
  %res = arith.select %eq, %c_eq, %gt_or_lt : !ord
  return %res : !ord
}

// CHECK-NOT: trait.proof private @PartialOrd_impl_i32_i32_p
trait.proof private @PartialOrd_impl_i32_i32_p {
  %d = trait.derive @PartialOrd[i32, i32] from @PartialOrd_impl_i32_i32 given()
  trait.return %d : !trait.claim<@PartialOrd[i32, i32]>
}

// CHECK-NOT: trait.proof private @Eq_impl_i32_p
trait.proof private @Eq_impl_i32_p {
  %d = trait.derive @Eq[i32] from @Eq_impl_i32 given()
  trait.return %d : !trait.claim<@Eq[i32]>
}

// CHECK-NOT: trait.proof private @Ord_impl_i32_p
trait.proof private @Ord_impl_i32_p {
  %d = trait.derive @Ord[i32] from @Ord_impl_i32 given()
  trait.return %d : !trait.claim<@Ord[i32]>
}

// CHECK-LABEL: func.func @max
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord.max_{{h[0-9a-f]+}}
func.func @max(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.func.call @Ord.max(%ord_p, %a, %b) : (!trait.claim<@Ord[i32] by @Ord_impl_i32_p>, i32, i32) -> i32

  return %res : i32
}

// CHECK-LABEL: func.func @min
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord.min_{{h[0-9a-f]+}}
func.func @min(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.func.call @Ord.min(%ord_p, %a, %b) : (!trait.claim<@Ord[i32] by @Ord_impl_i32_p>, i32, i32) -> i32

  return %res : i32
}
