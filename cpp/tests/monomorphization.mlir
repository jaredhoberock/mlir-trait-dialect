// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!S = !trait.poly<0>
!O = !trait.poly<1>
// CHECK-NOT: trait.trait private @PartialEq
trait.trait private @PartialEq(!trait.claim<@PartialEq[!S, !O]>) members [@PartialEq.eq, @PartialEq.neq]
func.func private @PartialEq.eq(!trait.claim<@PartialEq[!S, !O]>, !S, !O) -> i1
func.func private @PartialEq.neq(%self_claim: !trait.claim<@PartialEq[!S, !O]>, %self: !S, %other: !O) -> i1 {
  %equal = trait.func.call @PartialEq.eq(%self_claim, %self, %other) : (!trait.claim<@PartialEq[!S,!O]>, !S, !O) -> i1
  %true = arith.constant 1 : i1
  %res = arith.xori %equal, %true : i1
  return %res : i1
}

// CHECK-NOT: trait.impl private @PartialEq
trait.impl private @PartialEq_impl_i32_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) members [@PartialEq.eq -> @PartialEq_impl_i32_i32.eq] {
}
func.func private @PartialEq_impl_i32_i32.eq(%self_claim: !trait.claim<@PartialEq[i32, i32]>, %self: i32, %other: i32) -> i1 {
  %res = arith.cmpi eq, %self, %other : i32
  return %res : i1
}

!T = !trait.poly<2>

// CHECK-LABEL: func.func private @foo_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32.eq_{{h[0-9a-f]+}}
func.func private @foo(%c: !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, %x: !trait.poly<0>, %y: !trait.poly<0>) -> i1 {
  %res = trait.func.call @PartialEq.eq(%c, %x, %y) : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>,!trait.poly<0>) -> i1
  return %res : i1
}


// CHECK-LABEL: func.func @bar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @foo_{{.*}}
func.func @bar(%x: i32, %y: i32) -> i1 {
  %p = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  %res = trait.func.call @foo(%p, %x, %y)
    : (!trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>, i32, i32) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func private @baz_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32.eq_{{h[0-9a-f]+}}
// CHECK: call @PartialEq.neq_{{h[0-9a-f]+}}
func.func private @baz(%c: !trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, %x: !trait.poly<0>, %y: !trait.poly<0>) -> i1 {
  %eq = trait.func.call @PartialEq.eq(%c, %x, %y) : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>,!trait.poly<0>) -> i1

  %neq = trait.func.call @PartialEq.neq(%c, %x, %y) : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>,!trait.poly<0>) -> i1

  %res = arith.ori %eq, %neq : i1
  return %res : i1
}

// CHECK-LABEL: func.func @qux
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @baz_{{.*}}
func.func @qux(%x: i32, %y: i32) -> i1 {
  %p = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  %result = trait.func.call @baz(%p, %x, %y)
    : (!trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>, i32,i32) -> i1
  return %result : i1
}
