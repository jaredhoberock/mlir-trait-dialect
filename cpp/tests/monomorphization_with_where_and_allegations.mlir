// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!PartialEqS = !trait.poly<0>
!PartialEqO = !trait.poly<1>
// CHECK-NOT: trait.trait @PartialEq
trait.trait @PartialEq[!PartialEqS,!PartialEqO] {
  func.func private @eq(!PartialEqS, !PartialEqO) -> i1
  
  func.func @ne(%self: !PartialEqS, %other: !PartialEqO) -> i1 {
    %p = trait.assume @PartialEq[!PartialEqS,!PartialEqO]
    %equal = trait.method.call @PartialEq::@eq<%p>(%self, %other)
      : (!PartialEqS, !PartialEqO) -> i1
      as !trait.claim<@PartialEq[!PartialEqS,!PartialEqO]> (!PartialEqS,!PartialEqO) -> i1
    %true = arith.constant 1 : i1
    %not_equal = arith.xori %equal, %true : i1
    return %not_equal : i1
  }
}

// CHECK-NOT: trait.trait @PartialEq
trait.impl for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    return %equal : i1
  }
}

!T = !trait.poly<0>

// CHECK-LABEL: func.func @foo_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32_eq
func.func @foo(%w: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %res = trait.method.call @PartialEq::@eq<%w>(%x, %y)
    : (!PartialEqS, !PartialEqO) -> i1
    as !trait.claim<@PartialEq[!T,!T]> (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @bar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @foo_i32
func.func @bar(%x: i32, %y: i32) -> i1 {
  %w = trait.witness @PartialEq[i32,i32]
  %res = trait.func.call @foo(%w, %x, %y)
    : (!trait.claim<@PartialEq[!T,!T]>, !T, !T) -> i1
    as (!trait.claim<@PartialEq[i32,i32]>, i32, i32) -> i1

  return %res : i1
}

!EqS = !trait.poly<2>
// CHECK-NOT: @Eq
trait.trait @Eq[!EqS] given [
  @PartialEq[!EqS,!EqS]
]
{
}

// CHECK-NOT: @Eq
trait.impl for @Eq[i32] {}

// model Option<Ordering>
// 0: Less
// 1: Equal
// 2: Greater
// 3: None
!opt_ord = i2

!PartialOrdS = !trait.poly<3>
!PartialOrdO = !trait.poly<4>

// CHECK-NOT: trait.trait @PartialOrd
trait.trait @PartialOrd[!PartialOrdS,!PartialOrdO] given [
  @PartialEq[!PartialOrdS,!PartialOrdO]
]
{
  func.func private @partial_cmp(!PartialOrdS, !PartialOrdO) -> !opt_ord

  func.func @lt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %self_p = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %cmp = trait.method.call @PartialOrd::@partial_cmp<%self_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord
      as !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]> (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_lt = arith.constant 0 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_lt : !opt_ord
    return %res : i1
  }

  func.func @le(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %self_p = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %partial_eq_p = trait.project %self_p
      : @PartialOrd[!PartialOrdS,!PartialOrdO]
      to @PartialEq[!PartialOrdS,!PartialOrdO]

    %lt = trait.method.call @PartialOrd::@lt<%self_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1
      as !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]> (!PartialOrdS, !PartialOrdO) -> i1

    %eq = trait.method.call @PartialEq::@eq<%partial_eq_p>(%self, %other)
      : (!PartialEqS,!PartialEqO) -> i1
      as !trait.claim<@PartialEq[!PartialOrdS,!PartialOrdO]> (!PartialOrdS, !PartialOrdO) -> i1

    %res = arith.ori %lt, %eq : i1
    return %res : i1
  }

  func.func @gt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %self_p = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %cmp = trait.method.call @PartialOrd::@partial_cmp<%self_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord
      as !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]> (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_gt = arith.constant 2 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_gt : !opt_ord
    return %res : i1
  }

  func.func @ge(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %self_p = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %partial_eq_p = trait.project %self_p
      : @PartialOrd[!PartialOrdS,!PartialOrdO]
      to @PartialEq[!PartialOrdS,!PartialOrdO]

    %gt = trait.method.call @PartialOrd::@gt<%self_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1
      as !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]> (!PartialOrdS, !PartialOrdO) -> i1

    %eq = trait.method.call @PartialEq::@eq<%partial_eq_p>(%self, %other)
      : (!PartialEqS,!PartialEqO) -> i1
      as !trait.claim<@PartialEq[!PartialOrdS,!PartialOrdO]> (!PartialOrdS, !PartialOrdO) -> i1

    %res = arith.ori %gt, %eq : i1
    return %res : i1
  }
}

// CHECK-NOT: trait.impl @PartialOrd
trait.impl for @PartialOrd[i32,i32] {
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
trait.trait @Ord[!OrdS] given [
  @Eq[!OrdS],
  @PartialOrd[!OrdS,!OrdS]
]
{
  func.func private @cmp(!OrdS, !OrdS) -> !ord

  func.func @max(%self: !OrdS, %other: !OrdS) -> !OrdS {
    %self_p = trait.assume @Ord[!OrdS]
    %partial_ord_p = trait.project %self_p
      : @Ord[!OrdS]
      to @PartialOrd[!OrdS,!OrdS]

    %cond = trait.method.call @PartialOrd::@gt<%partial_ord_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1
      as !trait.claim<@PartialOrd[!OrdS,!OrdS]> (!OrdS,!OrdS) -> i1

    %res = scf.if %cond -> !OrdS {
      scf.yield %self : !OrdS
    } else {
      scf.yield %other : !OrdS
    }

    return %res : !OrdS
  }

  func.func @min(%self: !OrdS, %other: !OrdS) -> !OrdS {
    %self_p = trait.assume @Ord[!OrdS]
    %partial_ord_p = trait.project %self_p
      : @Ord[!OrdS]
      to @PartialOrd[!OrdS,!OrdS]

    %cond = trait.method.call @PartialOrd::@le<%partial_ord_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> i1
      as !trait.claim<@PartialOrd[!OrdS,!OrdS]> (!OrdS,!OrdS) -> i1

    %res = scf.if %cond -> !OrdS {
      scf.yield %self: !OrdS
    } else {
      scf.yield %other: !OrdS
    }

    return %res : !OrdS
  }
}

// CHECK-NOT: trait.impl @Ord
trait.impl for @Ord[i32] {
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

// CHECK-LABEL: func.func @max
// CHECK-NOT: trait.claim
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord_impl_i32_max
func.func @max(%a: i32, %b: i32) -> i32 {
  %p = trait.allege @Ord[i32]
  %res = trait.method.call @Ord::@max<%p>(%a, %b)
    : (!OrdS, !OrdS) -> !OrdS
    as !trait.claim<@Ord[i32]> (i32, i32) -> i32

  return %res : i32
}

// CHECK-LABEL: func.func @min
// CHECK-NOT: trait.claim
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @Ord_impl_i32_min
func.func @min(%a: i32, %b: i32) -> i32 {
  %p = trait.allege @Ord[i32]
  %res = trait.method.call @Ord::@min<%p>(%a, %b)
    : (!OrdS, !OrdS) -> !OrdS
    as !trait.claim<@Ord[i32]> (i32, i32) -> i32

  return %res : i32
}
