// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @X_all proves @X[T] out of @X[tuple<T>], and @p cites ITSELF for that
// obligation: read at @X[tuple<T>] the proof's declaration rebuilds it, so one
// level deep the citation carries and the proof verifies. Followed down, the
// derivation asks about @X[tuple<i32>], then @X[tuple<tuple<i32>>], and never
// ends. Every node is a new application, so the early exit on a bound
// obligation never fires; the number of obligations standing on the derivation
// is what stops it, exactly as it stops the same chain in impl selection.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@X[i32]
// CHECK: note: required by {{.*}}@X[tuple<i32>]
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @X[!trait.poly<0>] { func.func private @x() -> i64 }

trait.impl private @X_all for @X[!trait.poly<0>] where [@X[tuple<!trait.poly<0>>]] {
  func.func @x() -> i64 {
    %a = trait.assume @X[tuple<!trait.poly<0>>]
    %r = trait.method.call %a @X[tuple<!trait.poly<0>>]::@x() : () -> i64
    return %r : i64
  }
}

trait.proof private @p proves @X_all for @X[!trait.poly<0>] given [@p]

func.func @main() -> i64 {
  %w = trait.witness @p for @X[i32]
  %r = trait.method.call %w @X[i32]::@x() : () -> i64 by @p
  return %r : i64
}
