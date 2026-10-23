// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A claim carrying a type variable means something only inside a template,
// where an instance substitutes the variable. Outside one nothing ever does, so
// such a claim neither holds nor is proven, and erasure's entry refuses it
// rather than erase it unread. Instantiation proves and refuses monomorphic
// claims only, so the refusal is the same whether it ran first or not.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

func.func private @g(%e: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// A reflexive equality over a type variable, cited beside a leaf by a coerce
// whose own equality holds through the leaf and goes to a plain call.
// CHECK: error: equality '!trait.claim<tuple<i64, !trait.poly<0>> = tuple<i64, !trait.poly<0>>>' does not settle before erasure: it carries a type variable outside a template
// CHECK-NOT: error
func.func @use() -> i64 {
  %w = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64] : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %p = trait.witness refl : !trait.claim<tuple<i64, !trait.poly<0>> = tuple<i64, !trait.poly<0>>>
  %r = trait.witness refl : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
  %d = trait.coerce %r : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">> to !trait.claim<!trait.proj<@A[i64], "Item"> = i64> via (%w, %p) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<tuple<i64, !trait.poly<0>> = tuple<i64, !trait.poly<0>>>)
  %v = func.call @g(%d) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64
  return %v : i64
}

// -----

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

func.func private @g(%e: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// A false equality over a type variable, derived from a false hypothesis and
// cited only by a coerce whose own equality holds through its other leaf. The
// hypothesis is refused where it is assumed, and the refl and the derived
// equality for the variable they carry.
// CHECK: error: equality '!trait.claim<i32 = i64>' does not settle before erasure: its sides resolve to 'i32' and 'i64'
// CHECK: error: equality '!trait.claim<tuple<i64, !trait.poly<0>> = tuple<i64, !trait.poly<0>>>' does not settle before erasure: it carries a type variable outside a template
// CHECK: error: equality '!trait.claim<tuple<i64, !trait.poly<0>> = tuple<i32, !trait.poly<0>>>' does not settle before erasure: it carries a type variable outside a template
// CHECK-NOT: error
func.func @use(%h: !trait.claim<i32 = i64>) -> i64 {
  %w = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64] : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %r0 = trait.witness refl : !trait.claim<tuple<i64, !trait.poly<0>> = tuple<i64, !trait.poly<0>>>
  %c = trait.coerce %r0 : !trait.claim<tuple<i64, !trait.poly<0>> = tuple<i64, !trait.poly<0>>> to !trait.claim<tuple<i64, !trait.poly<0>> = tuple<i32, !trait.poly<0>>> via (%h) : (!trait.claim<i32 = i64>)
  %r1 = trait.witness refl : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
  %d = trait.coerce %r1 : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">> to !trait.claim<!trait.proj<@A[i64], "Item"> = i64> via (%w, %c) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<tuple<i64, !trait.poly<0>> = tuple<i32, !trait.poly<0>>>)
  %v = func.call @g(%d) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64
  return %v : i64
}

// -----

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) {}

// An allegation of an application over a type variable: instantiation proves
// monomorphic allegations only, so it stands to erasure, which refuses it.
// CHECK: error: claim '!trait.claim<@B[!trait.poly<0>]>' is not proven before erasure: it carries a type variable outside a template
func.func @alleged() -> i64 {
  %a = trait.allege @B[!trait.poly<0>]
  %c = arith.constant 7 : i64
  return %c : i64
}
