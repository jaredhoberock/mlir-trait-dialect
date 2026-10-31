// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' | FileCheck %s

// Matching a projection recurses through the trait-application type arguments
// it spells. The function implementing @Trait.method below specializes
// @Base[i32]::Assoc to i64. Its claim parameters contain two projections of @Fn::Output whose @Fn argument
// lists differ only by that nested spelling. @Base[i32]::Assoc is a sibling
// projection this impl's own bindings do not resolve, so the impl takes the
// equality as a where entry; the correspondence reads the trait's spelling
// through it, into the nested @Fn arguments, and the comparison made where the
// stage begins accepts the two written forms as the same signature.

!S = !trait.poly<0>
!F = !trait.poly<1>
!R = !trait.poly<2>

trait.trait private @Base(!trait.claim<@Base[!S]>) assoc {Assoc = 0}

trait.trait private @Fn(!trait.claim<@Fn[!trait.poly<0>, !trait.poly<1>]>) assoc {Output = 0}

trait.trait private @SameAs(!trait.claim<@SameAs[!trait.poly<0>, !trait.poly<1>]>)

trait.trait private @Trait(!trait.claim<@Trait[!S]>) -> !trait.claim<@Base[!S]> members [@Trait.method]
func.func private @Trait.method(!trait.claim<@Trait[!S]>, !S, !F, !trait.claim<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>]>, !trait.claim<@SameAs[
      !trait.proj<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>], "Output">,
      !trait.proj<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>], "Output">
    ]>) -> i32

trait.impl private @Base_i32(%self: !trait.claim<@Base[i32]>) bindings {Assoc = i64} {
}

trait.impl private @Trait_i32(%self_claim: !trait.claim<@Trait[i32]>, %assoc: !trait.claim<!trait.proj<@Base[i32], "Assoc"> = i64>) members [@Trait.method -> @Trait_i32.method] {
  %base = trait.witness @Base_i32 for @Base[i32]
  trait.return %base : !trait.claim<@Base[i32] by @Base_i32>
}
func.func private @Trait_i32.method(%self_claim: !trait.claim<@Trait[i32]>, %assoc: !trait.claim<!trait.proj<@Base[i32], "Assoc"> = i64>, %self: i32, %f: !trait.poly<0>, %fn: !trait.claim<@Fn[!trait.poly<0>, tuple<i64>]>, %same: !trait.claim<@SameAs[
      !trait.proj<@Fn[!trait.poly<0>, tuple<i64>], "Output">,
      !trait.proj<@Fn[!trait.poly<0>, tuple<i64>], "Output">
    ]>) -> i32 {
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}

// CHECK-LABEL: trait.impl private @Trait_i32(%self: !trait.claim<@Trait[i32]>, %assoc: !trait.claim<!trait.proj<@Base[i32], "Assoc"> = i64>) members [@Trait.method -> @Trait_i32.method]
// CHECK: func.func private @Trait_i32.method(
