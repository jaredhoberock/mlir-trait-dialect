// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait method returns a sibling trait's associated type whose only impl is
// CONDITIONAL. Substituting the impl's self application makes it the ground
// projection Sibling[i64]::Elem. The impl declaration boundary resolves it by
// module-visible impl lookup: the sole candidate impl is selected by head match
// alone, and its where-clause premise (@Needs[i64], which no impl proves) is
// NOT evaluated. A well-formed program discharges that head claim where the
// projection is spelled; the boundary trusts its producer and does not re-prove
// it, so the impl verifies.

!S = !trait.poly<0>

trait.trait @Needs[!S] {}

trait.trait @Sibling[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sibling_i64 for @Sibling[i64] where [@Needs[i64]] {
  trait.assoc_type @Elem = i64
}

trait.trait @Host[!S] {
  trait.assoc_type @Out
  func.func private @make(!S) -> !trait.proj<@Sibling[!S], "Elem">
}

// CHECK: trait.impl @Host_i64
trait.impl @Host_i64 for @Host[i64] {
  trait.assoc_type @Out = i64
  func.func @make(%x: i64) -> i64 {
    %r = ub.poison : i64
    return %r : i64
  }
}
