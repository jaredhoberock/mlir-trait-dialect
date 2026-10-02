// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run reads its impl's @Mark[i32] entry, which the receiver's proof @H
// discharges with @Seven (7), and passes it through a reflexive coerce, while
// its parameter's proof @W discharges @Wrapped's @Mark[i32] requirement with
// @Nine. The coerce's result carries its input's proof, whichever other
// position proves the same claim otherwise, so the call through it runs
// @Seven's method.

// CHECK: {{^}}7{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.trait private @Wrapped[!T] where [@Mark[!T]] {}
trait.trait private @Host[!T] {
  trait.method @run(!trait.claim<@Wrapped[!T]>) -> i64
}
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
trait.impl private @Wrapped_i32 for @Wrapped[i32] {}
trait.proof private @W proves @Wrapped_i32[] for @Wrapped[i32] given [@Nine]
trait.impl private @Host_i32 for @Host[i32] where [@Mark[i32]] {
  trait.method @run(%w: !trait.claim<@Wrapped[i32]>) -> i64 {
    %s = trait.assume 0 : !trait.claim<@Mark[i32]>
    %e = trait.witness refl : !trait.claim<i32 = i32>
    %m = trait.coerce %s : !trait.claim<@Mark[i32]> to !trait.claim<@Mark[i32]> via (%e) : (!trait.claim<i32 = i32>)
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @W>) -> i64 by @H
  return %v : i64
}
