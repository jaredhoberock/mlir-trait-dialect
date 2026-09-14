// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Two impls of @Foo bind @Foo[i32], so nothing resolves @Foo[i32]::Out. The
// obligation carrying that projection is @B's requirement read at @B[i32], and
// it is spelled in no operation -- the trait's where clause spells it over the
// trait's own variable -- so no walk over what the stage left behind can report
// it. Selection names the ambiguity at the refusal, where the call that raised
// the demand stands.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = i32 }
trait.impl private @Foo_i32 for @Foo[i32] { trait.assoc_type @Out = i32 }
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
trait.proof private @forged proves @B_i32 for @B[i32] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  // expected-error @below {{incoherent impls (multiple satisfiable) for '!trait.proj<@Foo[i32], "Out">'}}
  %r = trait.method.call %w @B[i32]::@value() : () -> i64 by @forged
  return %r : i64
}
