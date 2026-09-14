// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @P proves @B_blanket for @B[T], whose requirement is @A[Foo[T]::Out], and
// cites @A_i64 for it. @Foo has two conditional impls, so the impls alone
// settle Foo[T]::Out for nobody and the citation is left standing at @P's own
// claim. At i32 only @Foo_m applies and Foo[i32]::Out is i32, so the obligation
// is @A[i32], which @A_i64 does not discharge: dispatching @A's method through
// @P would run @A_i64's body where @A_i32's is the honest one. The stage reads
// the pair through what selection settled and refuses @P.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Other[!trait.poly<0>] {}
trait.impl private @Marker_any for @Marker[!trait.poly<0>] {}
trait.impl private @Other_i8 for @Other[i8] {}
trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_m for @Foo[!trait.poly<0>] where [@Marker[!trait.poly<0>]] { trait.assoc_type @Out = !trait.poly<0> }
trait.impl private @Foo_o for @Foo[!trait.poly<0>] where [@Other[!trait.poly<0>]] { trait.assoc_type @Out = i64 }
trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] { func.func private @b() -> i64 }
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
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  func.func @b() -> i64 {
    %s = trait.assume @B[!trait.poly<0>]
    %a = trait.project %s[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    return %r : i64
  }
}
// expected-error @below {{obligation '!trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>' of proof @P is discharged by no evidence}}
trait.proof private @P proves @B_blanket for @B[!trait.poly<0>] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @P for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @P
  return %r : i64
}
