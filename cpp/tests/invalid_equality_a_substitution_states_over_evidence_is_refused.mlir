// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @f states the equality of its parameter with itself and hands it to @h. A
// call supplying evidence as the parameter cuts instances whose equality
// relates claims: the clone substitutes the variable into the equality, which
// it can always build, and the verifier refuses what it built where it stands,
// at the instance of @h that takes it, as it refuses an equality over evidence
// written by hand (invalid_claim_inside_an_equality_endpoint.mlir).

trait.trait private @T(%s: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @I(%s: !trait.claim<@T[i32]>) {}
// expected-error @below {{a type-equality endpoint must not contain a claim: '!trait.claim<@T[i32]>' in '!trait.claim<!trait.claim<@T[i32]> = !trait.claim<@T[i32]>>'}}
func.func private @h(%e: !trait.claim<!trait.poly<0> = !trait.poly<0>>, %x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
func.func private @f(%x: !trait.poly<0>) -> !trait.poly<0> {
  %e = trait.witness refl : !trait.claim<!trait.poly<0> = !trait.poly<0>>
  %y = trait.func.call @h(%e, %x) : (!trait.claim<!trait.poly<0> = !trait.poly<0>>, !trait.poly<0>) -> !trait.poly<0>
  return %y : !trait.poly<0>
}
func.func @run() -> i64 {
  %w = trait.witness @I for @T[i32]
  %c = trait.coerce %w : !trait.claim<@T[i32] by @I> to !trait.claim<@T[i32]>
  %r = trait.func.call @f(%c) : (!trait.claim<@T[i32]>) -> !trait.claim<@T[i32]>
  %k = arith.constant 7 : i64
  return %k : i64
}
