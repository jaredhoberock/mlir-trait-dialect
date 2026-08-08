// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// The proven-claim respell sweep rewrites unproven application claims into
// their proven spellings by proof-memo lookup, so it runs only when the memo
// is non-empty. This module both records a proof (an impl resolves for
// @Bound[i64]) and carries an equality witness, so the sweep runs while an
// equality-arm claim is in the module. The sweep dispatches on the claim arm
// and leaves the equality claim alone; a claim of one arm carries no trait
// application for the other arm to read.

!S = !trait.poly<0>

trait.trait @Bound[!S] {
  func.func private @use(!S) -> i1
}
trait.impl @Bound_impl for @Bound[i64] {
  func.func @use(%self: i64) -> i1 {
    %t = arith.constant 1 : i1
    return %t : i1
  }
}

trait.trait @Assoc[!S] {
  trait.assoc_type @Output
}
trait.impl @Assoc_impl for @Assoc[i64] {
  trait.assoc_type @Output = i64
}

// CHECK-LABEL: func.func @carry
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.witness
// CHECK: return %arg0 : i64
func.func @carry(%x: i64) -> !trait.proj<@Assoc[i64], "Output"> {
  %b = trait.allege @Bound[i64]
  %used = trait.method.call %b @Bound[i64]::@use(%x) : (i64) -> i1
  %eq = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Output"> resolves i64 by @Assoc_impl
    : !trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>
  %c = trait.coerce %x : i64 to !trait.proj<@Assoc[i64], "Output">
    via (%eq) : (!trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>)
  return %c : !trait.proj<@Assoc[i64], "Output">
}
