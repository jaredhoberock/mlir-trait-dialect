// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @make_one and @make_two each return @Need at a projection spelling, derived
// from @Need_from over the @Bound[i64] evidence the body chose: @One (7) and
// @Two (9) respectively. A function's result claim repeats what its returns
// hand back and a call's result claim what its callee's signature says, so
// each call carries its body's proof, and the method called through it runs
// the chosen impl's.

// CHECK: {{^}}79{{$}}

!S = !trait.poly<0>
trait.trait private @Trait(%self: !trait.claim<@Trait[!S]>) {
  trait.assoc_type @Output
}
trait.impl private @Trait_impl(%self: !trait.claim<@Trait[i64]>) {
  trait.assoc_type @Output = i64
}
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
func.func private @make_one() -> !trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]> {
  %b = trait.witness @One for @Bound[i64]
  %eq = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[i64] by @One> to !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]> via (%eq) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
  %d = trait.derive @Need[!trait.proj<@Trait[i64], "Output">] from @Need_from[!trait.proj<@Trait[i64], "Output">] given(%c) : (!trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]>)
  return %d : !trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>
}
func.func private @make_two() -> !trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]> {
  %b = trait.witness @Two for @Bound[i64]
  %eq = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[i64] by @Two> to !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]> via (%eq) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
  %d = trait.derive @Need[!trait.proj<@Trait[i64], "Output">] from @Need_from[!trait.proj<@Trait[i64], "Output">] given(%c) : (!trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]>)
  return %d : !trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>
}
func.func @main() -> i64 {
  %one = func.call @make_one() : () -> !trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>
  %two = func.call @make_two() : () -> !trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>
  %a = trait.method.call %one @Need[!trait.proj<@Trait[i64], "Output">]::@get() : () -> i64
  %b = trait.method.call %two @Need[!trait.proj<@Trait[i64], "Output">]::@get() : () -> i64
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
