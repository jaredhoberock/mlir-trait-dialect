// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// An impl's member that forwards a call of the same generic trait member to
// another impl stands inside a template. The clone cut for the outer call is
// stamped under a substitution that binds the enclosing member's variable, and
// the nested call's instance is read off the operand and result types that
// substitution leaves: the forwarded call names the same member at the same
// argument the outer call was cut for, and both lower to plain calls.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<0>
!V = !trait.poly<9>

trait.trait private @Store(!trait.claim<@Store[!S]>) members [@Store.keep]
func.func private @Store.keep(!trait.claim<@Store[!S]>, !trait.poly<0>, !trait.poly<1>) -> !trait.poly<1>

trait.impl private @Store_impl_i64(%self_claim: !trait.claim<@Store[i64]>) members [@Store.keep -> @Store_impl_i64.keep] {
}
func.func private @Store_impl_i64.keep(%self_claim: !trait.claim<@Store[i64]>, %self: i64, %v: !trait.poly<0>) -> !trait.poly<0> {
  return %v : !trait.poly<0>
}

// Forwards to the i64 impl's member under the same member generic: the nested
// call spells this member's own variable in the argument position the trait
// member's !V stands in.
trait.impl private @Store_impl_i32(%self_claim: !trait.claim<@Store[i32]>) members [@Store.keep -> @Store_impl_i32.keep] {
}
func.func private @Store_impl_i32.keep(%self_claim: !trait.claim<@Store[i32]>, %self: i32, %v: !trait.poly<0>) -> !trait.poly<0> {
  %inner = arith.constant 0 : i64
  %p = trait.witness @Store_impl_i64 for @Store[i64]
  %r = trait.func.call @Store.keep(%p, %inner, %v) : (!trait.claim<@Store[i64] by @Store_impl_i64>, i64, !trait.poly<0>) -> !trait.poly<0>
  return %r : !trait.poly<0>
}

func.func @main(%x: i32, %v: f32) -> f32 {
  %p = trait.witness @Store_impl_i32 for @Store[i32]
  %r = trait.func.call @Store.keep(%p, %x, %v) : (!trait.claim<@Store[i32] by @Store_impl_i32>, i32, f32) -> f32
  return %r : f32
}

// The i64 member's instance for f32 stands beside its impl, then the forwarding
// member's instance calls it in place of the nested generic call, and main calls
// the forwarding instance.
// CHECK-LABEL: func.func private @Store_impl_i64.keep_{{h[0-9a-f]+}}
// CHECK-SAME: %{{.*}}: i64, %{{.*}}: f32) -> f32
// CHECK-LABEL: func.func private @Store_impl_i32.keep_{{h[0-9a-f]+}}
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: f32) -> f32
// CHECK-NOT: trait.func.call
// CHECK: call @Store_impl_i64.keep_{{h[0-9a-f]+}}
// CHECK-NOT: trait.func.call
// CHECK-LABEL: func.func @main
// CHECK: call @Store_impl_i32.keep_{{h[0-9a-f]+}}
