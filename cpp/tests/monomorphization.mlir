// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!S = !trait.poly<0>
!O = !trait.poly<1>
// CHECK-NOT: trait.trait @PartialEq
trait.trait @PartialEq [!S,!O] {
  func.func private @eq(!S, !O) -> i1

  func.func @neq(%self: !S, %other: !O) -> i1 {
    %partial_eq = trait.assume @PartialEq[!S,!O]
    %equal = trait.method.call %partial_eq @PartialEq[!S,!O]::@eq(%self, %other)
      : (!S, !O) -> i1
    %true = arith.constant 1 : i1
    %res = arith.xori %equal, %true : i1
    return %res : i1
  }
}

// CHECK-NOT: trait.impl @PartialEq
trait.impl @PartialEq_impl_i32_i32 for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

!T = !trait.poly<2>

// CHECK-LABEL: func.func @foo_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32_eq
func.func @foo(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
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

// CHECK-LABEL: func.func @baz_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @PartialEq_impl_i32_i32_eq
// CHECK: call @PartialEq_impl_i32_i32_neq
func.func @baz(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
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
