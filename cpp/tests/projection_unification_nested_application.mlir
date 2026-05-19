// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Projection unification must recurse through trait-application type arguments.
// The impl method below specializes @Base[i32]::Assoc to i64. Its where-clause
// claim contains two projections of @Fn::Output whose @Fn argument lists differ
// only by that nested spelling. The impl verifier should accept those as the
// same signature under the impl's associated-type binding.

!S = !trait.poly<0>
!F = !trait.poly<1>
!R = !trait.poly<2>

trait.trait @Base[!S] {
  trait.assoc_type @Assoc
}

trait.trait @Fn[!F, !R] {
  trait.assoc_type @Output
}

trait.trait @SameAs[!S, !R] {
}

trait.trait @Trait[!S] where [@Base[!S]] {
  func.func private @method(
    !S,
    !F,
    !trait.claim<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>]>,
    !trait.claim<@SameAs[
      !trait.proj<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>], "Output">,
      !trait.proj<@Fn[!F, tuple<!trait.proj<@Base[!S], "Assoc">>], "Output">
    ]>
  ) -> i32
}

trait.impl @Base_i32 for @Base[i32] {
  trait.assoc_type @Assoc = i64
}

trait.impl @Trait_i32 for @Trait[i32] {
  func.func @method(
    %self: i32,
    %f: !F,
    %fn: !trait.claim<@Fn[!F, tuple<i64>]>,
    %same: !trait.claim<@SameAs[
      !trait.proj<@Fn[!F, tuple<i64>], "Output">,
      !trait.proj<@Fn[!F, tuple<i64>], "Output">
    ]>
  ) -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}

// CHECK-LABEL: trait.impl @Trait_i32
// CHECK: func.func @method
