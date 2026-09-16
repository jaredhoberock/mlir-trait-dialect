// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// @f instantiated at T calls @g at tuple<T>, and @g at U calls @f at U, so
// every instance is cut from a template it has not been cut from before at a
// type one tuple larger. Nothing repeats, so no cycle guard sees it; what stops
// it is the count of instances of ONE template standing on the chain that
// reached the call. The refusal names the template and the ends of its chain.

// CHECK: 'trait.func.call' op reached the instantiation limit while instantiating @f: 128 instances of it stand on the chain that reaches this call
// CHECK: note: instantiated from @f
// CHECK: note: instantiated from @g
// CHECK: note: instantiated from @f
// CHECK: note: {{.*}} more frame(s) elided
// CHECK: note: instantiated from @g
// CHECK: note: instantiated from @f
// CHECK: note: instantiated from @g

func.func private @f(%x: !trait.poly<0>) -> i64 {
  %t = builtin.unrealized_conversion_cast %x : !trait.poly<0> to tuple<!trait.poly<0>>
  %r = trait.func.call @g(%t) : (tuple<!trait.poly<0>>) -> i64
  return %r : i64
}

func.func private @g(%y: !trait.poly<1>) -> i64 {
  %r = trait.func.call @f(%y) : (!trait.poly<1>) -> i64
  return %r : i64
}

func.func @main() -> i64 {
  %c = arith.constant 0 : i64
  %r = trait.func.call @f(%c) : (i64) -> i64
  return %r : i64
}
