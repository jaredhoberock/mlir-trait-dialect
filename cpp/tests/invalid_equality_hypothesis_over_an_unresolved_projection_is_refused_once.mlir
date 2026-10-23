// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A function assumes an equality whose endpoint names a projection no impl
// resolves. With a body, the signature's inputs are its entry block's
// arguments, so the equality stands at the block argument and is judged there
// once, with the projection's refusal beside it; the signature adds no second
// site. Without a body the signature is where it stands, and it is judged at
// the declaration. Instantiation owes nothing here, so the verdict is the same
// whether it ran first or not.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i64], "Item">'}}
// expected-error @below {{equality '!trait.claim<!trait.proj<@A[i64], "Item"> = i64>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func @use(%h: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}

// -----

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i64], "Item">'}}
// expected-error @below {{equality '!trait.claim<!trait.proj<@A[i64], "Item"> = i64>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func private @decl(!trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64
