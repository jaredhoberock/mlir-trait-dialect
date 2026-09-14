// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B's requirement is spelled over a projection, and two impls of @Foo bind
// @Foo[i32] -- both to i32 -- so the impls standing here resolve it for nobody
// and selection names that ambiguity.
// A citation read against an obligation nothing can settle is a citation
// nothing checked: @forged names @A_i64 where the projection denotes i32. The
// obligation is left undischarged instead, so @B_i32's requirement stands
// unproven and the call through it is refused rather than dispatched to
// @A_i64's method.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
// expected-note@+1 {{candidate}}
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = i32 }
// expected-note@+1 {{candidate}}
trait.impl private @Foo_i32 for @Foo[i32] { trait.assoc_type @Out = i32 }
trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] {
  func.func private @b(!trait.poly<0>) -> i64
}
trait.impl private @A_i32 for @A[i32] {
  func.func @a() -> i64 {
    %c = arith.constant 32 : i64
    return %c : i64
  }
}
trait.impl private @A_i64 for @A[i64] {
  func.func @a() -> i64 {
    %c = arith.constant 64 : i64
    return %c : i64
  }
}
trait.impl private @B_i32 for @B[i32] {
  func.func @b(%x: i32) -> i64 {
    %s = trait.assume @B[i32]
    // expected-error@+2 {{incoherent impls (multiple satisfiable) for '!trait.proj<@Foo[i32], "Out">'}}
    // expected-error@+1 {{unproven monomorphic claim '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>' after instantiate-monomorphs}}
    %a = trait.project %s[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[i32], "Out">]::@a() : () -> i64
    return %r : i64
  }
}
trait.proof private @forged proves @B_i32 for @B[i32] given [@A_i64]
func.func @main(%x: i32) -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b(%x) : (i32) -> i64 by @forged
  return %r : i64
}
