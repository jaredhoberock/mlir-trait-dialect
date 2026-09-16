// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// An impl's method that forwards a call of the same generic trait method to
// another impl stands inside a template. The clone cut for the outer call is
// stamped under a substitution that binds the enclosing method's variable, and
// the nested call's instance is read off the operand and result types that
// substitution leaves: the forwarded call names the same method at the same
// argument the outer call was cut for, and both lower to plain calls.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<0>
!V = !trait.poly<9>

trait.trait private @Store[!S] {
  func.func private @keep(!S, !V) -> !V
}

trait.impl private @Store_impl_i64 for @Store[i64] {
  func.func @keep(%self: i64, %v: !trait.poly<5>) -> !trait.poly<5> {
    return %v : !trait.poly<5>
  }
}

// Forwards to the i64 impl's method under the same method generic: the nested
// call spells this method's own variable in the argument position the trait
// method's !V stands in.
trait.impl private @Store_impl_i32 for @Store[i32] {
  func.func @keep(%self: i32, %v: !trait.poly<6>) -> !trait.poly<6> {
    %inner = arith.constant 0 : i64
    %p = trait.witness @Store_impl_i64 for @Store[i64]
    %r = trait.method.call %p @Store[i64]::@keep(%inner, %v)
      : (i64, !trait.poly<6>) -> !trait.poly<6>
      by @Store_impl_i64
    return %r : !trait.poly<6>
  }
}

func.func @main(%x: i32, %v: f32) -> f32 {
  %p = trait.witness @Store_impl_i32 for @Store[i32]
  %r = trait.method.call %p @Store[i32]::@keep(%x, %v)
    : (i32, f32) -> f32
    by @Store_impl_i32
  return %r : f32
}

// The i64 method's instance for f32 stands beside its impl, then the forwarding
// method's instance calls it in place of the nested generic call, and main calls
// the forwarding instance.
// CHECK-LABEL: func.func private @Store_impl_i64_keep_
// CHECK-SAME: %{{.*}}: i64, %{{.*}}: f32) -> f32
// CHECK-LABEL: func.func private @Store_impl_i32_keep_
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: f32) -> f32
// CHECK-NOT: trait.method.call
// CHECK: call @Store_impl_i64_keep_
// CHECK-NOT: trait.method.call
// CHECK-LABEL: func.func @main
// CHECK: call @Store_impl_i32_keep_
