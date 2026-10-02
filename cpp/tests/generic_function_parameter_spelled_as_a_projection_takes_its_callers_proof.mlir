// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @g's second parameter is spelled as the projection @Wrap[!T]::Item, which
// @Wrap_i32 binds to the claim @Mark[i32]. Whether a position takes evidence is
// read off its formal as the instance spells it, so the two calls -- one
// supplying @Mark[i32] through @One (7), one through @Two (9) -- name two
// instances of @g, and each runs the impl its own proof selects.

// CHECK: {{^}}79{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @One for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Two for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrap[!T] where [!trait.proj<@Wrap[!T], "Item"> = !trait.claim<@Mark[!T]>] {
  trait.assoc_type @Item
}
trait.impl private @Wrap_i32 for @Wrap[i32] {
  trait.assoc_type @Item = !trait.claim<@Mark[i32]>
}
func.func private @g(%x: !T, %c: !trait.proj<@Wrap[!T], "Item">, %w: !trait.claim<@Wrap[!T]>) -> i64 {
  %e = trait.project %w[0] : !trait.claim<@Wrap[!T]> -> !trait.claim<!trait.proj<@Wrap[!T], "Item"> = !trait.claim<@Mark[!T]>>
  %m = trait.coerce %c : !trait.proj<@Wrap[!T], "Item"> to !trait.claim<@Mark[!T]> via (%e) : (!trait.claim<!trait.proj<@Wrap[!T], "Item"> = !trait.claim<@Mark[!T]>>)
  %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %x = arith.constant 0 : i32
  %one = trait.witness @One for @Mark[i32]
  %two = trait.witness @Two for @Mark[i32]
  %w = trait.witness @Wrap_i32 for @Wrap[i32]
  %a = trait.func.call @g(%x, %one, %w) : (i32, !trait.claim<@Mark[i32] by @One>, !trait.claim<@Wrap[i32] by @Wrap_i32>) -> i64
  %b = trait.func.call @g(%x, %two, %w) : (i32, !trait.claim<@Mark[i32] by @Two>, !trait.claim<@Wrap[i32] by @Wrap_i32>) -> i64
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
