// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 >/dev/null | FileCheck %s --check-prefix=QUIET --allow-empty

// @Host_i32's method @m takes its second parameter spelled as the projection
// @Wrap[i32]::Item, which @Wrap_i32 binds to the claim @Mark[i32]. Whether a
// position takes evidence is read off its formal as the instance spells it, so
// the two calls -- one supplying @Mark[i32] through @One (7), one through @Two
// (9) -- name two instances of the method, and each runs the impl its own
// proof selects.

// CHECK: {{^}}79{{$}}
// A call whose evidence is not yet proven waits for it without a diagnostic.
// QUIET-NOT: error

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @One(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Two(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!T]>) -> !trait.claim<!trait.proj<@Wrap[!T], "Item"> = !trait.claim<@Mark[!T]>> {
  trait.assoc_type @Item
}
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) {
  trait.assoc_type @Item = !trait.claim<@Mark[i32]>
  %req0 = trait.allege !trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>
  trait.return %req0 : !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) {
  trait.method @m(!T, !trait.proj<@Wrap[!T], "Item">, !trait.claim<@Wrap[!T]>) -> i64
}
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>) {
  trait.method @m(%x: i32, %c: !trait.proj<@Wrap[i32], "Item">, %w: !trait.claim<@Wrap[i32]>) -> i64 {
    %e = trait.project %w[0] : !trait.claim<@Wrap[i32]> -> !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>
    %m = trait.coerce %c : !trait.proj<@Wrap[i32], "Item"> to !trait.claim<@Mark[i32]> via (%e) : (!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>)
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
func.func @main() -> i64 {
  %x = arith.constant 0 : i32
  %one = trait.witness @One for @Mark[i32]
  %two = trait.witness @Two for @Mark[i32]
  %w = trait.witness @Wrap_i32 for @Wrap[i32]
  %e = trait.witness proj_resolve !trait.proj<@Wrap[i32], "Item"> resolves !trait.claim<@Mark[i32]> by @Wrap_i32 : !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>
  %one_projected = trait.coerce %one : !trait.claim<@Mark[i32] by @One> to !trait.proj<@Wrap[i32], "Item"> via (%e) : (!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>)
  %two_projected = trait.coerce %two : !trait.claim<@Mark[i32] by @Two> to !trait.proj<@Wrap[i32], "Item"> via (%e) : (!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>)
  %h = trait.witness @Host_i32 for @Host[i32]
  %a = trait.method.call %h @Host[i32]::@m(%x, %one_projected, %w) : (i32, !trait.proj<@Wrap[i32], "Item">, !trait.claim<@Wrap[i32] by @Wrap_i32>) -> i64 by @Host_i32
  %b = trait.method.call %h @Host[i32]::@m(%x, %two_projected, %w) : (i32, !trait.proj<@Wrap[i32], "Item">, !trait.claim<@Wrap[i32] by @Wrap_i32>) -> i64 by @Host_i32
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
