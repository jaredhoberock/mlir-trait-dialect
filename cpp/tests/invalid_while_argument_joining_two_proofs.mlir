// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An scf.while enters with @Need[i64] derived over @One (7), and its after
// region hands back @Need[i64] derived over @Two (9). The before argument, the
// after argument and the loop's result join one another, and what enters them
// carries two proofs, so each has no type: each is refused where it stands,
// naming the two proofs that enter, and neither is chosen.

!S = !trait.poly<0>
trait.trait private @Bound(%self: !trait.claim<@Bound[!S]>) { trait.method @value() -> i64 }
trait.impl private @One(%self: !trait.claim<@Bound[i64]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Two(%self: !trait.claim<@Bound[i64]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Need(%self: !trait.claim<@Need[!S]>) { trait.method @get() -> i64 }
trait.impl private @Need_from(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) {
  trait.method @get() -> i64 {
    %v = trait.method.call %b @Bound[!S]::@value() : () -> i64
    trait.return %v : i64
  }
}
func.func private @pick() -> i64 {
  %b1 = trait.witness @One for @Bound[i64]
  %d1 = trait.derive @Need[i64] from @Need_from[i64] given(%b1) : (!trait.claim<@Bound[i64] by @One>)
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  // The loop's result and its before argument each stand on a line of their own.
  // expected-error @+4 {{unproven monomorphic claim '!trait.claim<@Need[i64]>' after instantiate-monomorphs}}
  // expected-note-re @+3 {{control flow joins it from '!trait.claim<@Need[i64] by @{{.*}}' and '!trait.claim<@Need[i64] by @{{.*}}'}}
  // expected-error @+3 {{unproven monomorphic claim '!trait.claim<@Need[i64]>' after instantiate-monomorphs}}
  // expected-note-re @+2 {{control flow joins it from '!trait.claim<@Need[i64] by @{{.*}}' and '!trait.claim<@Need[i64] by @{{.*}}'}}
  %r:2 = scf.while
      (%a = %d1, %n = %zero) : (!trait.claim<@Need[i64]>, i64) -> (!trait.claim<@Need[i64]>, i64) {
    %c = arith.cmpi slt, %n, %two : i64
    scf.condition(%c) %a, %n : !trait.claim<@Need[i64]>, i64
  } do {
  // expected-error @+2 {{unproven monomorphic claim '!trait.claim<@Need[i64]>' after instantiate-monomorphs}}
  // expected-note-re @+1 {{control flow joins it from '!trait.claim<@Need[i64] by @{{.*}}' and '!trait.claim<@Need[i64] by @{{.*}}'}}
  ^bb0(%h: !trait.claim<@Need[i64]>, %m: i64):
    %b2 = trait.witness @Two for @Bound[i64]
    %d2 = trait.derive @Need[i64] from @Need_from[i64] given(%b2) : (!trait.claim<@Bound[i64] by @Two>)
    %m1 = arith.addi %m, %one : i64
    scf.yield %d2, %m1 : !trait.claim<@Need[i64]>, i64
  }
  %v = trait.method.call %r#0 @Need[i64]::@get() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %v = func.call @pick() : () -> i64
  return %v : i64
}
