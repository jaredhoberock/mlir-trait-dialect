// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @X_all proves @X[T] out of @X[tuple<T>], and @p supplies ITSELF for that
// premise: read at @X[tuple<T>] the proof's declaration rebuilds it, so one
// level deep the citation carries and the proof verifies. Followed down, the
// derivation asks for @X[tuple<T>], then @X[tuple<tuple<T>>], and never ends.
// Every step is a new application, so no citation repeats; the depth of the
// obligation chain is what stops it, before any instance is cut, and each
// frame names the proof that put the next step on the chain.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@X[!trait.poly<0>]{{.*}}, stated by proof @p
// CHECK: note: required by {{.*}}@X[tuple<!trait.poly<0>>]{{.*}}, stated by proof @p
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @X(%self: !trait.claim<@X[!trait.poly<0>]>) { trait.method @x() -> i64 }

trait.impl private @X_all(%self: !trait.claim<@X[!trait.poly<0>]>, %x: !trait.claim<@X[tuple<!trait.poly<0>>]>) {
  trait.method @x() -> i64 {
    %r = trait.method.call %x @X[tuple<!trait.poly<0>>]::@x() : () -> i64
    trait.return %r : i64
  }
}

trait.proof private @p {
  %p0 = trait.witness @p for @X[tuple<!trait.poly<0>>]
  %d = trait.derive @X[!trait.poly<0>] from @X_all given(%p0) : (!trait.claim<@X[tuple<!trait.poly<0>>] by @p>)
  trait.return %d : !trait.claim<@X[!trait.poly<0>]>
}

func.func @main() -> i64 {
  %w = trait.witness @p for @X[i32]
  %r = trait.method.call %w @X[i32]::@x() : () -> i64 by @p
  return %r : i64
}
