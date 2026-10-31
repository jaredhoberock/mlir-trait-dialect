// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false %s 2>&1 | FileCheck %s

// Erasure removes a coerce to a claim whole and forwards a coerce of a value
// only where its endpoints coincide, so what a coerce relates must be settled
// before anything is erased: erasure's entry refuses, where they stand, an
// equality a coerce cites that does not hold, and a template still public.
// Each module runs erasure alone, as a pipeline running no instantiation hands
// it one.

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

trait.trait private @Bound(!trait.claim<@Bound[!trait.poly<0>]>)
trait.trait private @Assoc(!trait.claim<@Assoc[!trait.poly<0>]>) assoc {Output = 0}
trait.impl private @Assoc_impl(%self: !trait.claim<@Assoc[!trait.poly<0>]>) bindings {Output = !trait.poly<0>} {
}

// A claim-to-claim respell of an unproven argument, citing an equality that
// holds: a function with a body taking an unproven application claim is a
// template, cloned per proof, so the function is refused where it stands as a
// template still public, and the respell's endpoints are never compared.
// CHECK: :[[# @LINE + 2]]:1: error: 'func.func' op is a public template
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

trait.trait private @Bound(!trait.claim<@Bound[!trait.poly<0>]>)

// The unproven claim taken as an argument and passed straight to a plain call,
// with no coerce: each function is a template, the private one collected and
// the public one refused where it stands.
// CHECK: :[[# @LINE + 6]]:1: error: 'func.func' op is a public template
// CHECK-NOT: error
func.func private @g(%b: !trait.claim<@Bound[i64]>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}
func.func @use(%b: !trait.claim<@Bound[i64]>) -> i64 {
  %v = func.call @g(%b) : (!trait.claim<@Bound[i64]>) -> i64
  return %v : i64
}
