// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// requireProvenClaimOperands: the call @consume forwards the proj.cast result as
// an argument claim. That claim is unproven until the projection @Gen[i64]::A
// resolves and the cast folds, inheriting the input's proof. Specializing
// @consume against a still-unproven argument claim would bake an unprovable
// parameter into the clone, whose method call could never lower. The guard
// therefore defers call lowering until the operand claim is proven, making the
// result independent of the order proofs settle. Gate-neutering the guard lets
// @consume specialize early, and the cloned method call fails to legalize.

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

func.func @consume(%c: !trait.claim<@X[!trait.poly<1>]>) -> i64 {
  %r = trait.method.call %c @X[!trait.poly<1>]::@ping() : () -> i64
  return %r : i64
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: trait.func.call
// CHECK: call @consume
func.func @main() -> i64 {
  %x = trait.witness @X_i64 for @X[i64]
  %gen = trait.allege @Gen[i64]
  %cast = trait.proj.cast %x, %gen
    : !trait.claim<@X[i64] by @X_i64>
    to !trait.claim<@X[!trait.proj<@Gen[i64], "A">]>
    by !trait.claim<@Gen[i64]>
  %r = trait.func.call @consume(%cast) : (!trait.claim<@X[!trait.proj<@Gen[i64], "A">]>) -> i64
  return %r : i64
}
