// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// InheritProjCastProofPattern: once the projection @Gen[i64]::A in a cast's
// result resolves to i64, the cast has a proven input claim (@X[i64] by @X_i64)
// and an unproven result claim over the SAME trait application (@X[i64]). The
// pattern retypes the result to the input's proven claim so the now-identity
// cast folds away, and the method call through the cast result lowers to the
// concrete impl method. Gate-neutering the pattern leaves the result unproven,
// so the monomorphic claim survives instantiate-monomorphs and the run fails.

trait.trait @Gen[!trait.poly<0>] {
  trait.assoc_type @A
}

trait.impl @Gen_i64 for @Gen[i64] {
  trait.assoc_type @A = i64
}

trait.trait @X[!trait.poly<1>] {
  func.func private @ping() -> i64
}

trait.impl @X_i64 for @X[i64] {
  func.func @ping() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: trait.method.call
// CHECK: call @X_i64_ping
func.func @main() -> i64 {
  %x = trait.witness @X_i64 for @X[i64]
  %gen = trait.allege @Gen[i64]
  %cast = trait.proj.cast %x, %gen
    : !trait.claim<@X[i64] by @X_i64>
    to !trait.claim<@X[!trait.proj<@Gen[i64], "A">]>
    by !trait.claim<@Gen[i64]>
  %r = trait.method.call %cast @X[!trait.proj<@Gen[i64], "A">]::@ping() : () -> i64
  return %r : i64
}
