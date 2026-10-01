// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --check-prefix=INSTANCE

// @B's requirement projects through @Foo, which @B does not require, so no
// subproof of a proof of @B discharges @Foo and nothing at a known index of
// @forged says what that projection is. The proof op's verifier reads its
// citations through the evidence the proof holds and nothing else -- not the
// impls standing around it -- so it cannot decide this citation and declines
// it: the module verifies. The instance a call cuts reads the projection
// through the impl selection settles for @Foo[i32], and refuses the citation
// there.

// VERIFIED: trait.proof private @forged proves @B_blanket[!trait.poly<0> = !trait.poly<0>] for @B[!trait.poly<0>] given [@A_i64]
// INSTANCE: error: 'trait.method.call' op proof @A_i64 proves '!trait.claim<@A[i64]>', which does not discharge the obligation '!trait.claim<@A[i32]>'

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = !trait.poly<0> }
trait.trait private @A[!trait.poly<0>] { trait.method @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] { trait.method @b() -> i64 }
trait.impl private @A_i64 for @A[i64] {
  trait.method @a() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  trait.method @b() -> i64 {
    %s = trait.assume self : !trait.claim<@B[!trait.poly<0>]>
    %a = trait.project %s[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
}
trait.proof private @forged proves @B_blanket[!trait.poly<0> = !trait.poly<0>] for @B[!trait.poly<0>] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
