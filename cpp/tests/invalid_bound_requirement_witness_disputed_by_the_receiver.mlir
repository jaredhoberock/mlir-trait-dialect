// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @Wrapped states a bound requirement, @Mark of its associated type at every
// argument, and @W witnesses it by citing @Nine directly; @run projects the
// requirement off its parameter at i1, while the receiver's proof @PH
// discharges the impl's own @Mark[i32] entry with @Seven. No subproof names
// the bound requirement's proof, so selection is asked for it and meets two
// impls of @Mark[i32]: the program is refused rather than run through either.

// CHECK: error: incoherent impls (multiple satisfiable) for '!trait.claim<@Mark[i32]>'

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
trait.trait private @Wrapped[!T] where [forall [!trait.bound<0>] -> @Mark[!trait.proj<@Wrapped[!T], "Item", [!trait.bound<0>]>]] {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @W for @Wrapped[i32] witnesses [#trait<witness requirement 0 by @Nine>] {
  trait.assoc_type @Item<[!trait.poly<1>]> = i32
}
trait.proof private @PW proves @W[] for @Wrapped[i32] given [unit]
trait.trait private @Host[!T] { trait.method @run(!trait.claim<@Wrapped[!T]>) -> i64 }
trait.impl private @H for @Host[i32] where [@Mark[i32]] {
  trait.method @run(%w: !trait.claim<@Wrapped[i32]>) -> i64 {
    %m = trait.project %w[0] for [i1] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>]>
    %v = trait.method.call %m @Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @PH proves @H[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @PH for @Host[i32]
  %w = trait.witness @PW for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @PW>) -> i64 by @PH
  return %v : i64
}
