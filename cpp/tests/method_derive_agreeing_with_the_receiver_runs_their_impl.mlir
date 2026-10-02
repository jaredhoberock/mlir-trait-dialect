// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run derives @Wrapped[i32] from @Wrapped_i32 given @Mark[i32] projected off
// its parameter, whose proof @OP discharges it with @Nine (9); the receiver's
// proof @H supplies its own @Wrapped[i32] through @WR, whose premise is @Nine
// too. The proof of the derived claim discharges the impl's premise by the
// proof its operand names, so the derive is witnessed by it and the derived
// claim's method runs @Nine's although @Seven also implements @Mark[i32].

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Seven for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped[!T] { trait.method @value() -> i64 }
trait.impl private @Wrapped_i32 for @Wrapped[i32] where [@Mark[i32]] {
  trait.method @value() -> i64 {
    %m = trait.assume 0 : !trait.claim<@Mark[i32]>
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @WR proves @Wrapped_i32[] for @Wrapped[i32] given [@Nine]
trait.trait private @Outer[!T] where [@Mark[!T]] {}
trait.impl private @O for @Outer[i32] {}
trait.proof private @OP proves @O[] for @Outer[i32] given [@Nine]
trait.trait private @Host[!T] { trait.method @run(!trait.claim<@Outer[!T]>) -> i64 }
trait.impl private @Host_i32 for @Host[i32] where [@Wrapped[i32]] {
  trait.method @run(%c: !trait.claim<@Outer[i32]>) -> i64 {
    %m = trait.project %c[0] : !trait.claim<@Outer[i32]> -> !trait.claim<@Mark[i32]>
    %w = trait.derive @Wrapped[i32] from @Wrapped_i32[] given(%m) : (!trait.claim<@Mark[i32]>)
    %v = trait.method.call %w @Wrapped[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@WR]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %c = trait.witness @OP for @Outer[i32]
  %v = trait.method.call %h @Host[i32]::@run(%c) : (!trait.claim<@Outer[i32] by @OP>) -> i64 by @H
  return %v : i64
}
