// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Matching a projection recurses through the trait-application type arguments
// it spells. The impl method below specializes @Base[i32]::Assoc to i64. Its
// claim parameters contain two projections of @Fn::Output whose @Fn argument
// lists differ only by that nested spelling. @Base[i32]::Assoc is a sibling
// projection this impl's own bindings do not resolve, so the impl takes the
// equality as a where entry; the correspondence reads the trait's spelling
// through it, into the nested @Fn arguments, and accepts the two spellings as
// the same signature.

!S = !trait.poly<0>
!F = !trait.poly<1>
!R = !trait.poly<2>

trait.trait private @Base(%self: !trait.claim<@Base[!S]>) {
  trait.assoc_type @Assoc
}

trait.trait private @Fn(%self: !trait.claim<@Fn[!F, !R]>) {
  trait.assoc_type @Output
}

trait.trait private @SameAs(%self: !trait.claim<@SameAs[!S, !R]>) {
}

trait.trait private @Trait(%self: !trait.claim<@Trait[!S]>) -> !trait.claim<@Base[!S]> {
  trait.method @method(
    !S,
    !F,
    !trait.claim<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>]>,
    !trait.claim<@SameAs[
      !trait.proj<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>], "Output">,
      !trait.proj<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>], "Output">
    ]>
  ) -> i32
}

trait.impl private @Base_i32(%self: !trait.claim<@Base[i32]>) {
  trait.assoc_type @Assoc = i64
}

trait.impl private @Trait_i32(%self_claim: !trait.claim<@Trait[i32]>, %assoc: !trait.claim<!trait.proj<@Base[i32], "Assoc"> = i64>) {
  trait.method @method(
    %self: i32,
    %f: !F,
    %fn: !trait.claim<@Fn[!F, tuple<i64>]>,
    %same: !trait.claim<@SameAs[
      !trait.proj<@Fn[!F, tuple<i64>], "Output">,
      !trait.proj<@Fn[!F, tuple<i64>], "Output">
    ]>
  ) -> i32 {
    %c0 = arith.constant 0 : i32
    trait.return %c0 : i32
  }
  %base = trait.witness @Base_i32 for @Base[i32]
  trait.return %base : !trait.claim<@Base[i32] by @Base_i32>
}

// CHECK-LABEL: trait.impl private @Trait_i32(%self: !trait.claim<@Trait[i32]>, %assoc: !trait.claim<!trait.proj<@Base[i32], "Assoc"> = i64>)
// CHECK: trait.method @method
