// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!S = !trait.poly<0>
!O = !trait.poly<1>
// CHECK-NOT: trait.trait private @PartialEq
trait.trait private @PartialEq(%self_claim: !trait.claim<@PartialEq[!S, !O]>) {
  trait.method @eq(!S, !O) -> i1

  trait.method @neq(%self: !S, %other: !O) -> i1 {
    %equal = trait.method.call %self_claim @PartialEq[!S,!O]::@eq(%self, %other)
      : (!S, !O) -> i1
    %true = arith.constant 1 : i1
    %res = arith.xori %equal, %true : i1
    trait.return %res : i1
  }
}

// CHECK-NOT: trait.impl private @PartialEq
trait.impl private @PartialEq_impl_i32_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) {
  trait.method @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    trait.return %res : i1
  }
}

!T = !trait.poly<2>

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
  %p = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
  %res = trait.func.call @foo(%p, %x, %y)
    : (!trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>, i32, i32) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func private @baz_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32_{{h[0-9a-f]+}}_eq
// CHECK: call @PartialEq_{{h[0-9a-f]+}}_neq
func.func private @baz(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %eq = trait.method.call %c @PartialEq[!T,!T]::@eq(%x, %y)
    : (!T,!T) -> i1

  %neq = trait.method.call %c @PartialEq[!T,!T]::@neq(%x, %y)
    : (!T,!T) -> i1

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
