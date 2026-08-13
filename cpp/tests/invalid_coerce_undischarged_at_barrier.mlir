// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false %s 2>&1 | FileCheck %s

// A coerce whose endpoints still differ at the barrier is not discharged.
// Forwarding or dropping it would let a value keep a written type with no
// evidence left to justify it, so the barrier refuses it rather than cross it,
// and erase-polymorphs fails to legalize the op. This holds whether the coerce
// is used, dead, or a claim-to-claim respell whose projection never ground.

// A used coerce with divergent endpoints: forwarding its input would change the
// value's type unjustified.
// CHECK: failed to legalize operation 'trait.coerce'
func.func @undischarged(%v: i32, %e: !trait.claim<i32 = i16>) -> i16 {
  %c = trait.coerce %v : i32 to i16 via (%e) : (!trait.claim<i32 = i16>)
  return %c : i16
}

// -----

// A DEAD coerce with divergent endpoints. Dropping it silently would let an
// undischarged bridge disappear un-judged; the barrier judges its recorded
// endpoints and refuses.
// CHECK: failed to legalize operation 'trait.coerce'
func.func @dead_divergent(%v: i32, %e: !trait.claim<i32 = i16>) {
  %c = trait.coerce %v : i32 to i16 via (%e) : (!trait.claim<i32 = i16>)
  return
}

// -----

trait.trait @Bound[!trait.poly<0>] {}
trait.trait @Assoc[!trait.poly<0>] { trait.assoc_type @Output }

// A marked claim-to-claim respell, valid when the coerce verifies (its projection could
// converge), that reaches the barrier with the projection unresolved: its
// recorded endpoints still differ, so the claim-to-claim 1:0 erasure is refused
// rather than dropping an undischarged respell.
// CHECK: failed to legalize operation 'trait.coerce'
func.func @marked_respell_unresolved(
    %b: !trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>) {
  %c = trait.coerce %b : !trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>
    to !trait.claim<@Bound[i64]> unproven
  return
}
