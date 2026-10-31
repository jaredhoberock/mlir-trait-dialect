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
trait.trait private @Trait(!trait.claim<@Trait[!S]>) assoc {Output = 0}
trait.impl private @Trait_impl(%self: !trait.claim<@Trait[i64]>) bindings {Output = i64} {
}
trait.trait private @Bound(!trait.claim<@Bound[!S]>) members [@Bound.value]
func.func private @Bound.value(!trait.claim<@Bound[!S]>) -> i64
trait.impl private @One(%self: !trait.claim<@Bound[i64]>) members [@Bound.value -> @One.value] {
}
func.func private @One.value(%self: !trait.claim<@Bound[i64]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Two(%self: !trait.claim<@Bound[i64]>) members [@Bound.value -> @Two.value] {
}
func.func private @Two.value(%self: !trait.claim<@Bound[i64]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.trait private @Need(!trait.claim<@Need[!S]>) members [@Need.get]
func.func private @Need.get(!trait.claim<@Need[!S]>) -> i64
trait.impl private @Need_from(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) members [@Need.get -> @Need_from.get] {
}
func.func private @Need_from.get(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) -> i64 {
  %v = trait.func.call @Bound.value(%b) : (!trait.claim<@Bound[!S]>) -> i64
  return %v : i64
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
  %a = trait.func.call @Need.get(%one) : (!trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>) -> i64
  %b = trait.func.call @Need.get(%two) : (!trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>) -> i64
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
