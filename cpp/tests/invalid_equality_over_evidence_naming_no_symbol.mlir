// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An equality relating evidence over builtin types names no symbol, and the
// symbol-use walk reaches no type naming none, so the claim's symbol-use
// verifier never sees it (invalid_claim_inside_an_equality_endpoint.mlir
// holds the forms naming one). Every evidence op judges the evidence it reads
// and produces, at every verification: a coerce citing one, and the refl a
// substitution states one over, are refused there, so no coerce turns an i32
// into evidence and that evidence into an i64. One no
// evidence op reads, in a declaration's signature, is refused at erasure's
// entry, which judges every equality still standing.

// expected-error @below {{a predicate states types, and '!trait.claim<!trait.claim<i32 = i64> = i32>' resolves to '!trait.claim<i32 = i64>', which carries a claim}}
// expected-error @below {{equality '!trait.claim<!trait.claim<i32 = i64> = i32>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func private @f(!trait.claim<!trait.claim<i32 = i64> = i32>)

// -----

func.func @launder(%h: !trait.claim<!trait.claim<i32 = i64> = i32>, %x: i32) -> i64 {
  // expected-error @below {{'trait.coerce' op a type-equality endpoint must not contain a claim: '!trait.claim<i32 = i64>' in '!trait.claim<!trait.claim<i32 = i64> = i32>'}}
  %e = trait.coerce %x : i32 to !trait.claim<i32 = i64> via (%h) : (!trait.claim<!trait.claim<i32 = i64> = i32>)
  %y = trait.coerce %x : i32 to i64 via (%e) : (!trait.claim<i32 = i64>)
  return %y : i64
}

// -----

func.func private @h(%e: !trait.claim<!trait.poly<0> = !trait.poly<0>>, %x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
func.func private @g(%x: !trait.poly<0>) -> !trait.poly<0> {
  // expected-error @below {{'trait.witness' op a type-equality endpoint must not contain a claim: '!trait.claim<i32 = i32>' in '!trait.claim<!trait.claim<i32 = i32> = !trait.claim<i32 = i32>>'}}
  %e = trait.witness refl : !trait.claim<!trait.poly<0> = !trait.poly<0>>
  %y = trait.func.call @h(%e, %x) : (!trait.claim<!trait.poly<0> = !trait.poly<0>>, !trait.poly<0>) -> !trait.poly<0>
  return %y : !trait.poly<0>
}
func.func @run() -> i64 {
  %w = trait.witness refl : !trait.claim<i32 = i32>
  %r = trait.func.call @g(%w) : (!trait.claim<i32 = i32>) -> !trait.claim<i32 = i32>
  %k = arith.constant 7 : i64
  return %k : i64
}
