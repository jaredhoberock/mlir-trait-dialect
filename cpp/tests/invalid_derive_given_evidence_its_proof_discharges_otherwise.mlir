// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// @f derives @Wrapped[!T] from @W given its first parameter, which both calls
// supply as @Trait[i32] by @S1, and takes a second @Wrapped[!T] it never
// reads, @P in the first call, whose premise is @S0. The call supplies
// @Wrapped[i32] by @P alone, so the instance spells the derive with it; @P
// discharges @W's premise by @S0 where the derive is given @S1, so it is no
// proof of this derive, and the derive is refused rather than run through it.
// The instantiation stage that judges the derive fails on the refusal and
// reports nothing else.

// CHECK: error: 'trait.derive' op is given '!trait.claim<@Trait[i32] by @S1>' at where-clause entry 0, which @P discharges by @S0 instead
// CHECK-NOT: error:

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Trait[!T] { trait.method @method() -> i64 }
trait.impl private @S0 for @Trait[i32] {
  trait.method @method() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @S1 for @Trait[i32] {
  trait.method @method() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped[!T] { trait.method @method() -> i64 }
trait.impl private @W for @Wrapped[!U] where [@Trait[!U]] {
  trait.method @method() -> i64 {
    %c = trait.assume 0 : !trait.claim<@Trait[!U]>
    %v = trait.method.call %c @Trait[!U]::@method() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @P proves @W[!U = i32] for @Wrapped[i32] given [@S0]
trait.proof private @P9 proves @W[!U = i32] for @Wrapped[i32] given [@S1]
func.func private @f(%c: !trait.claim<@Trait[!T]>, %w: !trait.claim<@Wrapped[!T]>) -> i64 {
  %d = trait.derive @Wrapped[!T] from @W[!U = !T] given(%c) : (!trait.claim<@Trait[!T]>)
  %v = trait.method.call %d @Wrapped[!T]::@method() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %c = trait.witness @S1 for @Trait[i32]
  %w = trait.witness @P for @Wrapped[i32]
  %w9 = trait.witness @P9 for @Wrapped[i32]
  %a = trait.func.call @f(%c, %w) : (!trait.claim<@Trait[i32] by @S1>, !trait.claim<@Wrapped[i32] by @P>) -> i64
  %b = trait.func.call @f(%c, %w9) : (!trait.claim<@Trait[i32] by @S1>, !trait.claim<@Wrapped[i32] by @P9>) -> i64
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
