// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Test that the occurs check does not fire when an inference variable
// appears inside a projection type's trait application arguments.
//
// This models:
//   trait Callable { type Output; fn call(self) -> Self::Output; }
//   impl Callable for i64 { type Output = i64; ... }
//   fn apply<F: Callable>(f: F) -> F::Output { f.call() }
//   fn wrap_and_apply<T: Callable>(x: T) -> T::Output { apply(x) }
//
// When verifying the call to `apply` inside `wrap_and_apply`, after
// binding the type params, unification of the return types sees:
//   apply's return:          proj<@Callable[poly<10>], "Output">
//   wrap_and_apply's return: proj<@Callable[poly<20>], "Output">
// After instantiation, both poly vars map to inference vars.
// If wrap_and_apply's return were poly<20> (not a projection), the
// unifier would try: ?T := proj<Callable[?T], "Output"> — an occurs
// check that should pass because projections are opaque.

trait.trait @Callable[!trait.poly<0>] {
  trait.assoc_type @Output
  func.func private @call(!trait.poly<0>) -> !trait.proj<@Callable[!trait.poly<0>], "Output">
}

trait.impl @Callable_i64 for @Callable[i64] {
  trait.assoc_type @Output = i64
  func.func @call(%self: i64) -> i64 {
    return %self : i64
  }
}

// fn apply<F: Callable>(f: F, claim) -> Callable[F]::Output
func.func @apply(%f: !trait.poly<10>,
                 %claim: !trait.claim<@Callable[!trait.poly<10>]>)
    -> !trait.proj<@Callable[!trait.poly<10>], "Output"> {
  %r = trait.method.call %claim @Callable[!trait.poly<10>]::@call(%f)
    : (!trait.poly<10>) -> !trait.proj<@Callable[!trait.poly<10>], "Output">
  return %r : !trait.proj<@Callable[!trait.poly<10>], "Output">
}

// fn wrap_and_apply<T: Callable>(x: T, claim) -> Callable[T]::Output
func.func @wrap_and_apply(%x: !trait.poly<20>,
                          %claim: !trait.claim<@Callable[!trait.poly<20>]>)
    -> !trait.proj<@Callable[!trait.poly<20>], "Output"> {
  %r = trait.func.call @apply(%x, %claim)
      : (!trait.poly<20>,
         !trait.claim<@Callable[!trait.poly<20>]>)
      -> !trait.proj<@Callable[!trait.poly<20>], "Output">
  return %r : !trait.proj<@Callable[!trait.poly<20>], "Output">
}

// CHECK-LABEL: func.func @caller
// CHECK: call @wrap_and_apply_
// CHECK: return
func.func @caller() -> i64 {
  %x = arith.constant 42 : i64
  %w = trait.witness @Callable_i64 for @Callable[i64]
  %r = trait.func.call @wrap_and_apply(%x, %w)
      : (i64, !trait.claim<@Callable[i64] by @Callable_i64>)
      -> i64
  return %r : i64
}
