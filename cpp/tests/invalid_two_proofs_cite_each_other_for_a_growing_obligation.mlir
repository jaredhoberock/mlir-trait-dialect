// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @p cites @q and @q cites @p, over different applications: @P_all proves
// @P1[T] out of @Q1[T], and @Q_all proves @Q1[T] out of @P1[tuple<T>]. Each
// citation's declaration rebuilds the obligation it is read at, so both proofs
// verify one level deep; the derivation underneath them alternates traits and
// grows the type forever. What bounds it is the number of frames standing on
// the chain, whichever trait each names, so a derivation alternating two traits
// is refused at the same depth as one repeating a single trait.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@P1[i32]
// CHECK: note: required by {{.*}}@Q1[i32]
// CHECK: note: required by {{.*}}@P1[tuple<i32>]
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @P1[!trait.poly<0>] { func.func private @p() -> i64 }
trait.trait private @Q1[!trait.poly<0>] { func.func private @q() -> i64 }

trait.impl private @P_all for @P1[!trait.poly<0>] where [@Q1[!trait.poly<0>]] {
  func.func @p() -> i64 {
    %a = trait.assume @Q1[!trait.poly<0>]
    %r = trait.method.call %a @Q1[!trait.poly<0>]::@q() : () -> i64
    return %r : i64
  }
}

trait.impl private @Q_all for @Q1[!trait.poly<0>] where [@P1[tuple<!trait.poly<0>>]] {
  func.func @q() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}

trait.proof private @p proves @P_all for @P1[!trait.poly<0>] given [@q]
trait.proof private @q proves @Q_all for @Q1[!trait.poly<0>] given [@p]

func.func @main() -> i64 {
  %w = trait.witness @p for @P1[i32]
  %r = trait.method.call %w @P1[i32]::@p() : () -> i64 by @p
  return %r : i64
}
