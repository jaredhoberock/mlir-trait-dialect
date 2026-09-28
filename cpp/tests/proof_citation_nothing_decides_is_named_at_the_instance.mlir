// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --check-prefix=INSTANCE

// @B's requirement projects through @Foo, and no impl of @Foo exists, so
// nothing decides what @forged's citation of @A_i64 discharges: the proof's
// verifier declines it and the module verifies. At the instance the hop off
// the proven claim is left unproven, selection cannot prove it either, and the
// refusal names the citation that nothing decided.

// VERIFIED: trait.proof private @forged proves @B_blanket for @B[!trait.poly<0>] given [@A_i64]
// INSTANCE: error: unproven monomorphic claim '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>' after instantiate-monomorphs
// INSTANCE: note: proof @forged cites @A_i64 for requirement 0, which nothing decides here

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] { func.func private @b() -> i64 }
trait.impl private @A_i64 for @A[i64] {
  func.func @a() -> i64 {
    %c = arith.constant 64 : i64
    return %c : i64
  }
}
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  func.func @b() -> i64 {
    %s = trait.assume @B[!trait.poly<0>]
    %a = trait.project %s[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    return %r : i64
  }
}
trait.proof private @forged proves @B_blanket for @B[!trait.poly<0>] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
