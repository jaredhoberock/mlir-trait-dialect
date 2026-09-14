// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Pins how a projection over a type variable crosses a call: the formal and the
// actual project one associated type over two different labels, and the
// comparison recurses through the projection's trait-application arguments, so
// the label the callee spells takes the one the caller supplies.
//
// This models:
//   trait Callable { type Output; fn call(self) -> Self::Output; }
//   impl Callable for i64 { type Output = i64; ... }
//   fn apply<F: Callable>(f: F) -> F::Output { f.call() }
//   fn wrap_and_apply<T: Callable>(x: T) -> T::Output { apply(x) }
//
// Verifying the call to `apply` inside `wrap_and_apply` compares:
//   apply's return:          proj<@Callable[poly<10>], "Output">
//   wrap_and_apply's return: proj<@Callable[poly<20>], "Output">
// The call's type arguments substitute poly<20> for poly<10>, and the two
// spellings then agree. (A projection meeting a rigid type it cannot resolve is
// a strict mismatch under the module-free comparison a verifier uses; the
// module-capable entry a pass or a committed-fact build uses resolves it where
// a unique impl binds it.)

trait.trait private @Callable[!trait.poly<0>] {
  trait.assoc_type @Output
  func.func private @call(!trait.poly<0>) -> !trait.proj<@Callable[!trait.poly<0>], "Output">
}

trait.impl private @Callable_i64 for @Callable[i64] {
  trait.assoc_type @Output = i64
  func.func @call(%self: i64) -> i64 {
    return %self : i64
  }
}

// fn apply<F: Callable>(f: F, claim) -> Callable[F]::Output
func.func private @apply(%f: !trait.poly<10>,
                 %claim: !trait.claim<@Callable[!trait.poly<10>]>)
    -> !trait.proj<@Callable[!trait.poly<10>], "Output"> {
  %r = trait.method.call %claim @Callable[!trait.poly<10>]::@call(%f)
    : (!trait.poly<10>) -> !trait.proj<@Callable[!trait.poly<10>], "Output">
  return %r : !trait.proj<@Callable[!trait.poly<10>], "Output">
}

// fn wrap_and_apply<T: Callable>(x: T, claim) -> Callable[T]::Output
func.func private @wrap_and_apply(%x: !trait.poly<20>,
                          %claim: !trait.claim<@Callable[!trait.poly<20>]>)
    -> !trait.proj<@Callable[!trait.poly<20>], "Output"> {
  %r = trait.func.call @apply(%x, %claim) {type_params = [!trait.poly<10>], type_args = [!trait.poly<20>]}
      : (!trait.poly<20>,
         !trait.claim<@Callable[!trait.poly<20>]>)
      -> !trait.proj<@Callable[!trait.poly<20>], "Output">
  return %r : !trait.proj<@Callable[!trait.poly<20>], "Output">
}

// The call keeps @wrap_and_apply's declared @Callable::Output projection
// spelling for its result; monomorphization resolves the projection to i64.
// CHECK-LABEL: func.func @caller
// CHECK: call @wrap_and_apply_
// CHECK: return
func.func @caller() -> !trait.proj<@Callable[i64], "Output"> {
  %x = arith.constant 42 : i64
  %w = trait.witness @Callable_i64 for @Callable[i64]
  %r = trait.func.call @wrap_and_apply(%x, %w) {type_params = [!trait.poly<20>], type_args = [i64]}
      : (i64, !trait.claim<@Callable[i64] by @Callable_i64>)
      -> !trait.proj<@Callable[i64], "Output">
  return %r : !trait.proj<@Callable[i64], "Output">
}
