// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// An scf.for carries @Need[i64] derived over @Two, though @One proves
// @Bound[i64] too, and its body hands the iteration argument back through an
// scf.if both of whose arms yield it. The iteration argument, the inner
// result and the loop's result join only one another and the init, so they
// carry the init's proof and the method called through the loop's result runs
// @Two's; selection, which would meet two impls, is never asked for them.

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
func.func private @pick(%flag: i1) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %b = trait.witness @Two for @Bound[i64]
  %d = trait.derive @Need[i64] from @Need_from[i64] given(%b) : (!trait.claim<@Bound[i64] by @Two>)
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%a = %d) -> (!trait.claim<@Need[i64]>) {
    %x = scf.if %flag -> !trait.claim<@Need[i64]> {
      scf.yield %a : !trait.claim<@Need[i64]>
    } else {
      scf.yield %a : !trait.claim<@Need[i64]>
    }
    scf.yield %x : !trait.claim<@Need[i64]>
  }
  %v = trait.method.call %r @Need[i64]::@get() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %f = arith.constant false
  %v = func.call @pick(%f) : (i1) -> i64
  return %v : i64
}
