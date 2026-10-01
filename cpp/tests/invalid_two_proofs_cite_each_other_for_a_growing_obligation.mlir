// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @p cites @q and @q cites @p, over different applications: @P_all proves
// @P1[T] out of @Q1[T], and @Q_all proves @Q1[T] out of @P1[tuple<T>]. Each
// citation's declaration rebuilds the obligation it is read at, so both proofs
// verify one level deep; the derivation underneath them alternates traits and
// grows the type forever. What bounds it is the number of frames standing on
// the chain, whichever trait each names, so a derivation alternating two traits
// is refused at the same depth as one repeating a single trait. Each frame
// names the proof whose citation put the next one on the chain, which is how
// the two alternating declarations are told apart.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@P1[i32]{{.*}}, stated by proof @p
// CHECK: note: required by {{.*}}@Q1[i32]{{.*}}, stated by proof @q
// CHECK: note: required by {{.*}}@P1[tuple<i32>]{{.*}}, stated by proof @p
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @P1[!trait.poly<0>] { trait.method @p() -> i64 }
trait.trait private @Q1[!trait.poly<0>] { trait.method @q() -> i64 }

trait.impl private @P_all for @P1[!trait.poly<0>] where [@Q1[!trait.poly<0>]] {
  trait.method @p() -> i64 {
    %a = trait.assume 0 : !trait.claim<@Q1[!trait.poly<0>]>
    %r = trait.method.call %a @Q1[!trait.poly<0>]::@q() : () -> i64
    trait.return %r : i64
  }
}

trait.impl private @Q_all for @Q1[!trait.poly<0>] where [@P1[tuple<!trait.poly<0>>]] {
  trait.method @q() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

trait.proof private @p proves @P_all[!trait.poly<0> = !trait.poly<0>] for @P1[!trait.poly<0>] given [@q]
trait.proof private @q proves @Q_all[!trait.poly<0> = !trait.poly<0>] for @Q1[!trait.poly<0>] given [@p]

func.func @main() -> i64 {
  %w = trait.witness @p for @P1[i32]
  %r = trait.method.call %w @P1[i32]::@p() : () -> i64 by @p
  return %r : i64
}
