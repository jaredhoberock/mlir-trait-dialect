// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!PartialEqS = !trait.poly<0>
!PartialEqO = !trait.poly<1>
// CHECK-NOT: trait.trait @PartialEq
trait.trait @PartialEq[!PartialEqS,!PartialEqO] {
  func.func private @eq(!PartialEqS, !PartialEqO) -> i1
  
  func.func @ne(%self: !PartialEqS, %other: !PartialEqO) -> i1 {
    %partial_eq = trait.assume @PartialEq[!PartialEqS,!PartialEqO]
    %equal = trait.method.call %partial_eq @PartialEq[!PartialEqS,!PartialEqO]::@eq(%self, %other)
      : (!PartialEqS, !PartialEqO) -> i1
    %true = arith.constant 1 : i1
    %not_equal = arith.xori %equal, %true : i1
    return %not_equal : i1
  }
}

// CHECK-NOT: trait.impl @PartialEq
trait.impl @PartialEq_impl_i32_i32 for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    return %equal : i1
  }
}

!T = !trait.poly<0>

// CHECK-LABEL: func.func @foo_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32_eq
func.func @foo(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %res = trait.method.call %c @PartialEq[!T,!T]::@eq(%x, %y)
    : (!T, !T) -> i1
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
trait.trait @Eq[!EqS] where [
  @PartialEq[!EqS,!EqS]
]
{
}

// CHECK-NOT: trait.impl @Eq
trait.impl @Eq_impl_i32 for @Eq[i32] {}

// model Option<Ordering>
// 0: Less
// 1: Equal
// 2: Greater
// 3: None
!opt_ord = i2

!PartialOrdS = !trait.poly<3>
!PartialOrdO = !trait.poly<4>

// CHECK-NOT: trait.trait @PartialOrd
trait.trait @PartialOrd[!PartialOrdS,!PartialOrdO] where [
  @PartialEq[!PartialOrdS,!PartialOrdO]
]
{
  func.func private @partial_cmp(!PartialOrdS, !PartialOrdO) -> !opt_ord

  func.func @lt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %partial_ord = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %cmp = trait.method.call %partial_ord @PartialOrd[!PartialOrdS,!PartialOrdO]::@partial_cmp(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_lt = arith.constant 0 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_lt : !opt_ord
    return %res : i1
  }

  func.func @le(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %partial_ord = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %partial_eq = trait.project %partial_ord
      : @PartialOrd[!PartialOrdS,!PartialOrdO]
      to @PartialEq[!PartialOrdS,!PartialOrdO]

    %lt = trait.method.call %partial_ord @PartialOrd[!PartialOrdS,!PartialOrdO]::@lt(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1

    %eq = trait.method.call %partial_eq @PartialEq[!PartialOrdS,!PartialOrdO]::@eq(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1

    %res = arith.ori %lt, %eq : i1
    return %res : i1
  }

  func.func @gt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %partial_ord = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %cmp = trait.method.call %partial_ord @PartialOrd[!PartialOrdS,!PartialOrdO]::@partial_cmp(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_gt = arith.constant 2 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_gt : !opt_ord
    return %res : i1
  }

  func.func @ge(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %partial_ord = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %partial_eq = trait.project %partial_ord
      : @PartialOrd[!PartialOrdS,!PartialOrdO]
      to @PartialEq[!PartialOrdS,!PartialOrdO]

    %gt = trait.method.call %partial_ord @PartialOrd[!PartialOrdS,!PartialOrdO]::@gt(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1

    %eq = trait.method.call %partial_eq @PartialEq[!PartialOrdS,!PartialOrdO]::@eq(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1

    %res = arith.ori %gt, %eq : i1
    return %res : i1
  }
}

// CHECK-NOT: trait.impl @PartialOrd
trait.impl @PartialOrd_impl_i32_i32 for @PartialOrd[i32,i32] {
  func.func @partial_cmp(%a: i32, %b: i32) -> !opt_ord {
    %c_lt = arith.constant 0 : !opt_ord
    %c_eq = arith.constant 1 : !opt_ord
    %c_gt = arith.constant 2 : !opt_ord

    %lt = arith.cmpi slt, %a, %b : i32
    %eq = arith.cmpi eq,  %a, %b : i32
    %gt_or_lt = arith.select %lt, %c_lt, %c_gt : !opt_ord
    %res = arith.select %eq, %c_eq, %gt_or_lt : !opt_ord
    return %res : !opt_ord
  }
}

// model Ordering
// 0: Less
// 1: Equal
// 2: Greater
!ord = i2

!OrdS = !trait.poly<5>
// CHECK-NOT: trait.trait @Ord
trait.trait @Ord[!OrdS] where [
  @Eq[!OrdS],
  @PartialOrd[!OrdS,!OrdS]
]
{
  func.func private @cmp(!OrdS, !OrdS) -> !ord

  func.func @max(%self: !OrdS, %other: !OrdS) -> !OrdS {
    %ord = trait.assume @Ord[!OrdS]
    %partial_ord = trait.project %ord
      : @Ord[!OrdS]
      to @PartialOrd[!OrdS,!OrdS]

    %cond = trait.method.call %partial_ord @PartialOrd[!OrdS,!OrdS]::@gt(%self, %other)
      : (!OrdS,!OrdS) -> i1

    %res = scf.if %cond -> !OrdS {
      scf.yield %self : !OrdS
    } else {
      scf.yield %other : !OrdS
    }

    return %res : !OrdS
  }

  func.func @min(%self: !OrdS, %other: !OrdS) -> !OrdS {
    %ord = trait.assume @Ord[!OrdS]
    %partial_ord = trait.project %ord
      : @Ord[!OrdS]
      to @PartialOrd[!OrdS,!OrdS]

    %cond = trait.method.call %partial_ord @PartialOrd[!OrdS,!OrdS]::@le(%self, %other)
      : (!OrdS,!OrdS) -> i1

    %res = scf.if %cond -> !OrdS {
      scf.yield %self: !OrdS
    } else {
      scf.yield %other: !OrdS
    }

    return %res : !OrdS
  }
}

// CHECK-NOT: trait.impl @Ord
trait.impl @Ord_impl_i32 for @Ord[i32] {
  func.func @cmp(%a: i32, %b: i32) -> !ord {
    %lt = arith.cmpi slt, %a, %b : i32
    %eq = arith.cmpi eq,  %a, %b : i32

    %c_lt = arith.constant 0 : !ord
    %c_eq = arith.constant 1 : !ord
    %c_gt = arith.constant 2 : !ord

    %gt_or_lt = arith.select %lt, %c_lt, %c_gt : !ord
    %res = arith.select %eq, %c_eq, %gt_or_lt : !ord
    return %res : !ord
  }
}

// CHECK-NOT: trait.proof @PartialOrd_impl_i32_i32_p
trait.proof @PartialOrd_impl_i32_i32_p proves @PartialOrd_impl_i32_i32 for @PartialOrd[i32,i32] given [
  @PartialEq_impl_i32_i32
]

// CHECK-NOT: trait.proof @Eq_impl_i32_p
trait.proof @Eq_impl_i32_p proves @Eq_impl_i32 for @Eq[i32] given [
  @PartialEq_impl_i32_i32
]

// CHECK-NOT: trait.proof @Ord_impl_i32_p
trait.proof @Ord_impl_i32_p proves @Ord_impl_i32 for @Ord[i32] given [
  @Eq_impl_i32_p,
  @PartialOrd_impl_i32_i32_p
]

// CHECK-LABEL: func.func @max
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord_impl_i32_max
func.func @max(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.method.call %ord_p @Ord[i32]::@max(%a, %b)
    : (i32, i32) -> i32
    by @Ord_impl_i32_p

  return %res : i32
}

// CHECK-LABEL: func.func @min
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord_impl_i32_min
func.func @min(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.method.call %ord_p @Ord[i32]::@min(%a, %b)
    : (i32, i32) -> i32
    by @Ord_impl_i32_p

  return %res : i32
}
