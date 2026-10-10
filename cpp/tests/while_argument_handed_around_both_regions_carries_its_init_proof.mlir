// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// An scf.while carries @Need[i64] derived over @Two, though @One proves
// @Bound[i64] too, through two iterations, its before region forwarding the
// argument to its after region and its after region handing it back. The
// before argument, the after argument and the loop's result join only one
// another and the init, so they carry the init's proof and the method called
// through the loop's result runs @Two's; selection, which would meet two
// impls, is never asked for them.

// CHECK: {{^}}9{{$}}

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
  %b = trait.witness @Two for @Bound[i64]
  %d = trait.derive @Need[i64] from @Need_from[i64] given(%b) : (!trait.claim<@Bound[i64] by @Two>)
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  %r:2 = scf.while (%a = %d, %n = %zero) : (!trait.claim<@Need[i64]>, i64) -> (!trait.claim<@Need[i64]>, i64) {
    %c = arith.cmpi slt, %n, %two : i64
    scf.condition(%c) %a, %n : !trait.claim<@Need[i64]>, i64
  } do {
  ^bb0(%h: !trait.claim<@Need[i64]>, %m: i64):
    %m1 = arith.addi %m, %one : i64
    scf.yield %h, %m1 : !trait.claim<@Need[i64]>, i64
  }
  %v = trait.method.call %r#0 @Need[i64]::@get() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %v = func.call @pick() : () -> i64
  return %v : i64
}
