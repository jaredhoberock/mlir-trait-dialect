// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B's requirement projects through @Foo, which @Foo_any binds to the trait's
// own variable, so @B_blanket's obligation at an instance is @A of that
// instance. @forged cites @A_i64 for it. At the declaration the obligation
// still spells a projection over a variable and nothing decides it; the
// witness names @B[i32], where the obligation reads as @A[i32] and the
// citation is read there.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = !trait.poly<0> }
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
trait.proof private @forged proves @B_blanket for @B[!trait.poly<0>] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  // expected-error @below {{proof @A_i64 proves '!trait.claim<@A[i64]>', which does not discharge the obligation '!trait.claim<@A[i32]>'}}
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
