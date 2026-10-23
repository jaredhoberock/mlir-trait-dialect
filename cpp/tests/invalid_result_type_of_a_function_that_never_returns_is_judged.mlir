// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A function that never returns hands back no value of its result type, so its
// declared result type stands in its signature alone, and it is judged there
// as every standing claim is before erasure removes it: a proof cited over a
// projection no impl serves, and an equality whose sides are two types.
// result_type_of_a_function_that_never_returns_resolves.mlir is the twin an
// impl of @A for i64 makes true.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @I(%self: !trait.claim<@B[!trait.proj<@A[i64], "Item">]>) {}

// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i64], "Item">'}}
// expected-error @below {{claim '!trait.claim<@B[!trait.proj<@A[i64], "Item">] by @I>' is not proven before erasure: an argument it states resolves to no type a predicate states}}
func.func @f() -> !trait.claim<@B[!trait.proj<@A[i64], "Item">] by @I> {
  cf.br ^loop
^loop:
  cf.br ^loop
}

// -----

// expected-error @below {{equality '!trait.claim<i32 = i64>' does not settle before erasure: its sides resolve to 'i32' and 'i64'}}
func.func @f() -> !trait.claim<i32 = i64> {
  cf.br ^loop
^loop:
  cf.br ^loop
}
