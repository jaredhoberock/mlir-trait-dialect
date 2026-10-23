// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B_proj proves @B at a projection of @A[i64], and no impl of @A serves i64,
// so the claim its witness carries states a proposition about no type. Nothing
// here is work instantiation owes -- the claim is proven by hand, and a claim's
// arguments are what it states, never an obligation of their own -- so the
// lowering driver hands the module to erasure alone. Erasure's entry reads a
// proven claim's application as selection reads every application before it
// proves one, whatever ran before it, and refuses one whose argument resolves
// to no type where it stands rather than erase it unread: at the witness, and
// at the parameter that takes it; selection's refusal of the projection is
// named once, at the first.
// projection_in_a_proven_claims_argument_settles_at_erasure.mlir is the twin an
// impl of @A for i64 makes true.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_proj(%self: !trait.claim<@B[!trait.proj<@A[i64], "Item">]>) {}

// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i64], "Item">'}}
// expected-error @below {{claim '!trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>' is not proven before erasure: an argument it states resolves to no type a predicate states}}
func.func @use(%w: !trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

func.func @run() -> i64 {
  // expected-error @below {{claim '!trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>' is not proven before erasure: an argument it states resolves to no type a predicate states}}
  %w = trait.witness @B_proj for @B[!trait.proj<@A[i64], "Item">]
  %v = func.call @use(%w) : (!trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>) -> i64
  return %v : i64
}
