// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Resolving the independent @Res[i64]::X projection cannot remove ambiguity
// between two @Gen[i64] impls. The @Gen[i64]::A projection remains an error.

!T = !trait.poly<0>

trait.trait private @Gen[!T] {
  trait.assoc_type @A
}

trait.impl private @Gen_wide for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.impl private @Gen_narrow for @Gen[i64] {
  trait.assoc_type @A = i16
}

trait.trait private @Res[!T] {
  trait.assoc_type @X
}

trait.impl private @Res_i64 for @Res[i64] {
  trait.assoc_type @X = i32
}

func.func @main() -> (!trait.proj<@Gen[i64], "A">, !trait.proj<@Res[i64], "X">) {
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Gen[i64], "A">
  %s = ub.poison : !trait.proj<@Res[i64], "X">
  return %r, %s : !trait.proj<@Gen[i64], "A">, !trait.proj<@Res[i64], "X">
}
