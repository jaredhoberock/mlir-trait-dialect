// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A trait method returns a SIBLING trait's associated type. Substituting the
// impl's self application turns that into a ground projection
// (Sibling[i64]::Elem) that this impl's own bindings do not resolve. The impl
// declares a premise citing @Sibling_i64, so the verifier replays it (to i32)
// before the strict comparison, and an impl method returning a DIFFERENT rigid
// type is rejected: the resolved sibling grade meets the impl's rigid return
// and the two spellings differ.

!S = !trait.poly<0>

trait.trait @Sibling[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sibling_i64 for @Sibling[i64] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  trait.assoc_type @Out
  func.func private @make(!S) -> !trait.proj<@Sibling[!S], "Elem">
}

// expected-error @below {{type mismatch: expected 'i32' but found 'i64'}}
// expected-error @below {{has incompatible signature}}
trait.impl @Host_i64 for @Host[i64]
    witnesses [#trait<witness !trait.proj<@Sibling[i64], "Elem"> = i32 by @Sibling_i64>] {
  trait.assoc_type @Out = i64
  // The sibling projection resolves to i32, but this method returns i64.
  func.func @make(%x: i64) -> i64 {
    %r = ub.poison : i64
    return %r : i64
  }
}
