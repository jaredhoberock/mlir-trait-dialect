// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: test everything

// CHECK-LABEL: trait @PartialEq [!trait.poly<0>, !trait.poly<1>]
!PartialEqS = !trait.poly<0>
!PartialEqO = !trait.poly<1>
trait.trait @PartialEq[!PartialEqS,!PartialEqO] {
  // CHECK-LABEL: func.func private @eq
  func.func private @eq(!PartialEqS, !PartialEqO) -> i1
  
  // CHECK-LABEL: func.func @ne
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

// CHECK-LABEL: trait.impl for @PartialEq[i32, i32]
trait.impl for @PartialEq[i32,i32] {
  // CHECK-LABEL: func @eq
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    return %equal : i1
  }
}

// CHECK-LABEL: func.func @foo
!T = !trait.poly<0>
!W = !trait.claim<@PartialEq[!T,!T]>
func.func @foo(%w: !W, %x: !T, %y: !T) -> i1 {
  // CHECK: %[[RES:.*]] = trait.method.call @PartialEq
  %res = trait.method.call @PartialEq::@eq<%w>(%x, %y)
    : (!PartialEqS, !PartialEqO) -> i1
    as !W (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @bar
func.func @bar(%x: i32, %y: i32) -> i1 {
  %w = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]

  // CHECK: %[[RES:.*]] = trait.func.call @foo
  %res = trait.func.call @foo(%w, %x, %y)
    : (!W,!T,!T) -> i1
    as (!trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>, i32, i32) -> i1

  return %res : i1
}

// CHECK-LABEL: trait @Eq [!trait.poly<2>]
!EqS = !trait.poly<2>
trait.trait @Eq[!EqS] where [
  @PartialEq[!EqS,!EqS]
]
{
}

// CHECK-LABEL: impl for @Eq[i32]
trait.impl for @Eq[i32] {}

// model Option<Ordering>
// 0: Less
// 1: Equal
// 2: Greater
// 3: None
!opt_ord = i2

// CHECK-LABEL: trait @PartialOrd [!trait.poly<3>, !trait.poly<4>] where [@PartialEq
!PartialOrdS = !trait.poly<3>
!PartialOrdO = !trait.poly<4>
trait.trait @PartialOrd[!PartialOrdS,!PartialOrdO] where [
  @PartialEq[!PartialOrdS,!PartialOrdO]
]
{
  // CHECK-LABEL: func.func private @partial_cmp
  func.func private @partial_cmp(!PartialOrdS, !PartialOrdO) -> !opt_ord

  // CHECK-LABEL: func.func @lt
  func.func @lt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %self_p = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %cmp = trait.method.call @PartialOrd::@partial_cmp<%self_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord
      as !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]> (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_lt = arith.constant 0 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_lt : !opt_ord
    return %res : i1
  }

  // CHECK-LABEL: func.func @le
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

  // CHECK-LABEL: func.func @gt
  func.func @gt(%self: !PartialOrdS, %other: !PartialOrdO) -> i1 {
    %self_p = trait.assume @PartialOrd[!PartialOrdS,!PartialOrdO]

    %cmp = trait.method.call @PartialOrd::@partial_cmp<%self_p>(%self, %other)
      : (!PartialOrdS,!PartialOrdO) -> !opt_ord
      as !trait.claim<@PartialOrd[!PartialOrdS,!PartialOrdO]> (!PartialOrdS,!PartialOrdO) -> !opt_ord

    %ord_gt = arith.constant 2 : !opt_ord
    %res = arith.cmpi eq, %cmp, %ord_gt : !opt_ord
    return %res : i1
  }

  // CHECK-LABEL: func.func @ge
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

// CHECK-LABEL: trait.impl for @PartialOrd[i32, i32]
trait.impl for @PartialOrd[i32,i32] {
  // CHECK-LABEL: func.func @partial_cmp
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

// CHECK-LABEL: trait @Ord [!trait.poly<5>] where [@Eq[!trait.poly<5>], @PartialOrd[!trait.poly<5>, !trait.poly<5>]
!OrdS = !trait.poly<5>
trait.trait @Ord[!OrdS] where [
  @Eq[!OrdS],
  @PartialOrd[!OrdS,!OrdS]
]
{
  // CHECK-LABEL: func.func private @cmp
  func.func private @cmp(!OrdS, !OrdS) -> !ord

  // CHECK-LABEL: func.func @max
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

  // CHECK-LABEL: func.func @min
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

// CHECK-LABEL: trait.impl for @Ord[i32]
trait.impl for @Ord[i32] {
  // CHECK-LABEL: func.func @cmp
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

// CHECK-LABEL: trait.proof @PartialOrd_impl_i32_i32_p
trait.proof @PartialOrd_impl_i32_i32_p proves @PartialOrd_impl_i32_i32 for @PartialOrd[i32,i32] given [
  @PartialEq_impl_i32_i32
]

// CHECK-LABEL: trait.proof @Eq_impl_i32_p
trait.proof @Eq_impl_i32_p proves @Eq_impl_i32 for @Eq[i32] given [
  @PartialEq_impl_i32_i32
]

// CHECK-LABEL: trait.proof @Ord_impl_i32_p
trait.proof @Ord_impl_i32_p proves @Ord_impl_i32 for @Ord[i32] given [
  @Eq_impl_i32_p,
  @PartialOrd_impl_i32_i32_p
]

// CHECK-LABEL: func.func @max
func.func @max(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.method.call @Ord::@max<%ord_p>(%a, %b)
    : (!OrdS, !OrdS) -> !OrdS
    as !trait.claim<@Ord[i32] by @Ord_impl_i32_p> (i32, i32) -> i32

  return %res : i32
}

// CHECK-LABEL: func.func @min
func.func @min(%a: i32, %b: i32) -> i32 {
  %ord_p = trait.witness @Ord_impl_i32_p for @Ord[i32]

  %res = trait.method.call @Ord::@min<%ord_p>(%a, %b)
    : (!OrdS, !OrdS) -> !OrdS
    as !trait.claim<@Ord[i32] by @Ord_impl_i32_p> (i32, i32) -> i32

  return %res : i32
}
