// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait method returns a sibling trait's associated type whose only impl is
// CONDITIONAL. Substituting the impl's self application makes it the ground
// projection Sibling[i64]::Elem. The host impl declares a premise citing that
// conditional impl; verification is obligation-aware, so the premise is legal only
// because the host impl's own where-clause carries @Needs[i64], which discharges
// the cited impl's assumption. The verifier never enumerates candidates or
// trusts an unproven head claim -- it replays the verified premise and accepts.

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
trait.impl @Host_i64 for @Host[i64] where [@Needs[i64]]
    premises [#trait<witness !trait.proj<@Sibling[i64], "Elem"> = i64 by @Sibling_i64>] {
  trait.assoc_type @Out = i64
  func.func @make(%x: i64) -> i64 {
    %r = ub.poison : i64
    return %r : i64
  }
}
