// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Two impls of @Foo bind @Foo[i32], so nothing resolves @Foo[i32]::Out. The
// claim carrying that projection is @B's requirement read at @B[i32]: @B_i32
// returns an allegation of it, which replaces the projection off @B[i32]'s
// proof. Selection refuses the allegation where the impl wrote it, and the
// stage's exit walk names the claim it leaves unproven there and, beside it,
// the candidates that make the projection its predicate spells ambiguous.
//
// The stage fails on the refusal, so the steps after it never run on a module
// nothing proved. The second run reads the exit status, which the diagnostic
// verifier does not.

// CHECK: error: incoherent impls (multiple satisfiable) for '!trait.proj<@Foo[i32], "Out">'
// CHECK: note: candidate
// CHECK: note: candidate

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
// expected-note@+1 {{candidate}}
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out = i32 }
// expected-note@+1 {{candidate}}
trait.impl private @Foo_i32(%self: !trait.claim<@Foo[i32]>) { trait.assoc_type @Out = i32 }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {
  trait.method @a() -> i64
}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> {
  trait.method @value() -> i64
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @value() -> i64 {
    %c = arith.constant 13 : i64
    trait.return %c : i64
  }
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>' after instantiate-monomorphs}}
  // expected-error @below {{incoherent impls (multiple satisfiable) for '!trait.proj<@Foo[i32], "Out">'}}
  %req0 = trait.allege @A[!trait.proj<@Foo[i32], "Out">]
  trait.return %req0 : !trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>
}
trait.proof private @forged {
  %d = trait.derive @B[i32] from @B_i32 given()
  trait.return %d : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %a = trait.project %w[0] : !trait.claim<@B[i32] by @forged> -> !trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>
  %r = trait.method.call %a @A[!trait.proj<@Foo[i32], "Out">]::@a() : () -> i64
  return %r : i64
}
