// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false %s 2>&1 | FileCheck %s

// Erasure removes a coerce to a claim whole and forwards a coerce of a value
// only where its endpoints coincide, so what a coerce relates must be settled
// before anything is erased: erasure's entry refuses, where they stand, an
// equality a coerce cites that does not hold and an application claim nothing
// proved. Each module runs erasure alone, as a pipeline running no
// instantiation hands it one; the lowering driver would run instantiation
// first on the last two, whose unproven claims are pending, and refuse them
// there.

// A used coerce of a value citing a false hypothesis: the hypothesis is refused
// where the function assumes it, before any coerce is erased.
// CHECK: error: equality '!trait.claim<i32 = i16>' does not settle before erasure: its sides resolve to 'i32' and 'i16'
func.func @undischarged(%v: i32, %e: !trait.claim<i32 = i16>) -> i16 {
  %c = trait.coerce %v : i32 to i16 via (%e) : (!trait.claim<i32 = i16>)
  return %c : i16
}

// -----

// A dead coerce of a value citing the same is refused alike.
// CHECK: error: equality '!trait.claim<i32 = i16>' does not settle before erasure: its sides resolve to 'i32' and 'i16'
func.func @dead_divergent(%v: i32, %e: !trait.claim<i32 = i16>) {
  %c = trait.coerce %v : i32 to i16 via (%e) : (!trait.claim<i32 = i16>)
  return
}

// -----

trait.trait private @Bound(%self: !trait.claim<@Bound[!trait.poly<0>]>) {}
trait.trait private @Assoc(%self: !trait.claim<@Assoc[!trait.poly<0>]>) { trait.assoc_type @Output }
trait.impl private @Assoc_impl(%self: !trait.claim<@Assoc[!trait.poly<0>]>) { trait.assoc_type @Output = !trait.poly<0> }

// A claim-to-claim respell of an unproven argument, citing an equality that
// holds: the argument is refused as an unproven claim where the function
// assumes it, and the respell's endpoints are never compared.
// CHECK: error: unproven monomorphic claim '!trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>' before erasure
// CHECK-NOT: error
func.func @respell_unresolved(
    %b: !trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>,
    %e: !trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>) {
  %c = trait.coerce %b : !trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>
    to !trait.claim<@Bound[i64]>
    via (%e) : (!trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>)
  return
}

// -----

trait.trait private @Bound(%self: !trait.claim<@Bound[!trait.poly<0>]>) {}

// The unproven claim taken as an argument and passed straight to a plain call,
// with no coerce: refused where each function assumes it.
// CHECK: error: unproven monomorphic claim '!trait.claim<@Bound[i64]>' before erasure
// CHECK: error: unproven monomorphic claim '!trait.claim<@Bound[i64]>' before erasure
func.func private @g(%b: !trait.claim<@Bound[i64]>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}
func.func @use(%b: !trait.claim<@Bound[i64]>) -> i64 {
  %v = func.call @g(%b) : (!trait.claim<@Bound[i64]>) -> i64
  return %v : i64
}
