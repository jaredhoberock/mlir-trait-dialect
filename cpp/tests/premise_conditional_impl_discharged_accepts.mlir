// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A premise citing a conditional impl is legal exactly when the citing impl's
// own where-clause covers the cited impl's assumptions. @Sib_i64_cond binds
// Sib[i64]::Elem = i32 and assumes @X[i64]; @Host_i64 assumes @X[i64] too, so
// the audit discharges the cited impl's assumption against the citing impl's
// where entry, replays the premise, and the expected signature reduces to
// (i64) -> i32 and matches.

!S = !trait.poly<0>

trait.trait @X[!S] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64_cond for @Sib[i64] where [@X[i64]] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// CHECK: trait.impl @Host_i64 for @Host[i64]where [@X[i64]]premises [#trait<certificate!trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond>]
trait.impl @Host_i64 for @Host[i64] where [@X[i64]]
    premises [#trait<certificate !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
