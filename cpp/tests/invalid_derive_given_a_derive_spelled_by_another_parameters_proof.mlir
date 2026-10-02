// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// @f derives @Wrapped from @W given its parameter %a, then @Outer from @O
// given that derive. The call supplies %a by @Nine and %u by @O7, whose tree
// proves @Wrapped[i32] by @W7 over @Seven; @Mark[i32], proven two ways, binds
// nothing, and @Wrapped[i32] is spelled @W7, so the inner derive is spelled
// @W7 and the outer one @O7, which agree with each other. The outer derive is
// judged only once the derive it is given is, and that one is given @Nine
// where @W7 discharges its entry by @Seven: it is refused rather than
// confirming the outer derive and dying with it.
// The instantiation stage that judges the derive fails on the refusal and
// reports nothing else.

// CHECK: error: 'trait.derive' op is given '!trait.claim<@Mark[i32] by @Nine>' at where-clause entry 0, which @W7 discharges by @Seven instead
// CHECK-NOT: error:

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
trait.impl private @W for @Wrapped[!T] where [@Mark[!T]] {
  trait.method @value() -> i64 {
    %p = trait.assume 0 : !trait.claim<@Mark[!T]>
    %v = trait.method.call %p @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.trait private @Outer[!T] { trait.method @value() -> i64 }
trait.impl private @O for @Outer[!T] where [@Wrapped[!T]] {
  trait.method @value() -> i64 {
    %p = trait.assume 0 : !trait.claim<@Wrapped[!T]>
    %v = trait.method.call %p @Wrapped[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @W7 proves @W[!T = i32] for @Wrapped[i32] given [@Seven]
trait.proof private @O7 proves @O[!T = i32] for @Outer[i32] given [@W7]
func.func private @f(%a: !trait.claim<@Mark[!T]>, %u: !trait.claim<@Outer[!T]>) -> i64 {
  %w = trait.derive @Wrapped[!T] from @W[!T = !T] given(%a) : (!trait.claim<@Mark[!T]>)
  %o = trait.derive @Outer[!T] from @O[!T = !T] given(%w) : (!trait.claim<@Wrapped[!T]>)
  %v = trait.method.call %o @Outer[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %a = trait.witness @Nine for @Mark[i32]
  %u = trait.witness @O7 for @Outer[i32]
  %v = trait.func.call @f(%a, %u) : (!trait.claim<@Mark[i32] by @Nine>, !trait.claim<@Outer[i32] by @O7>) -> i64
  return %v : i64
}
