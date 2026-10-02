// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @p cites @q and @q cites @p, over different applications: @P_all proves
// @P1[T] out of @Q1[T], and @Q_all proves @Q1[T] out of @P1[tuple<T>]. Each
// proof verifies against the other's declaration one level deep. Followed
// down, the derivation alternates the two traits and the application it asks
// about grows forever; the depth of the obligation chain bounds it, and each
// frame names the proof that put the next step on the chain, which is how the
// two alternating declarations are told apart.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@P1[!trait.poly<0>]{{.*}}, stated by proof @p
// CHECK: note: required by {{.*}}@Q1[!trait.poly<0>]{{.*}}, stated by proof @q
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @P1(%self: !trait.claim<@P1[!trait.poly<0>]>) { trait.method @p() -> i64 }
trait.trait private @Q1(%self: !trait.claim<@Q1[!trait.poly<0>]>) { trait.method @q() -> i64 }

trait.impl private @P_all(%self: !trait.claim<@P1[!trait.poly<0>]>, %q1: !trait.claim<@Q1[!trait.poly<0>]>) {
  trait.method @p() -> i64 {
    %r = trait.method.call %q1 @Q1[!trait.poly<0>]::@q() : () -> i64
    trait.return %r : i64
  }
}

trait.impl private @Q_all(%self: !trait.claim<@Q1[!trait.poly<0>]>, %p1: !trait.claim<@P1[tuple<!trait.poly<0>>]>) {
  trait.method @q() -> i64 {
    %r = trait.method.call %p1 @P1[tuple<!trait.poly<0>>]::@p() : () -> i64
    trait.return %r : i64
  }
}

trait.proof private @p {
  %p0 = trait.witness @q for @Q1[!trait.poly<0>]
  %d = trait.derive @P1[!trait.poly<0>] from @P_all given(%p0) : (!trait.claim<@Q1[!trait.poly<0>] by @q>)
  trait.return %d : !trait.claim<@P1[!trait.poly<0>]>
}
trait.proof private @q {
  %p0 = trait.witness @p for @P1[tuple<!trait.poly<0>>]
  %d = trait.derive @Q1[!trait.poly<0>] from @Q_all given(%p0) : (!trait.claim<@P1[tuple<!trait.poly<0>>] by @p>)
  trait.return %d : !trait.claim<@Q1[!trait.poly<0>]>
}

func.func @main() -> i64 {
  %w = trait.witness @p for @P1[i32]
  %r = trait.method.call %w @P1[i32]::@p() : () -> i64 by @p
  return %r : i64
}
