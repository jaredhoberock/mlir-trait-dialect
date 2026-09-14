// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B's requirement at @B[i32] is @A[Foo[i32]::Out], and nothing implements @Foo
// at i32, so the projection is settled by nothing. @forged cites @A_i64 for it.
// At @forged's own claim the citation is neither carried nor refused -- the
// obligation still spells a projection -- and the obligation is spelled in no
// operation, so no walk over what the stage left standing finds it. The stage
// reads the pair once more through what impl selection settled, which is still
// nothing, and refuses the proof.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_i8 for @Foo[i8] { trait.assoc_type @Out = i8 }
trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] where [@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] {
  func.func private @value() -> i64
}
trait.impl private @A_i64 for @A[i64] {}
trait.impl private @B_i32 for @B[i32] {
  func.func @value() -> i64 {
    %c = arith.constant 13 : i64
    return %c : i64
  }
}
// expected-error @below {{obligation '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>' of proof @forged is discharged by no evidence}}
trait.proof private @forged proves @B_i32 for @B[i32] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@value() : () -> i64 by @forged
  return %r : i64
}
