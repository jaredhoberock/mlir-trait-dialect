// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B's second requirement projects through @Foo, its first. Two impls of @Foo
// stand at i32, binding Out to i32 and to i64, so the impls around the proof
// do not say what @Foo[i32]::Out is; the subproof @forged names at the index of
// @Foo[i32] does. Read through it, the obligation is @A[i64], and @forged
// cites @A_i32 for it.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = !trait.poly<0> }
trait.impl private @Foo_i32 for @Foo[i32] { trait.assoc_type @Out = i64 }
trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@Foo[!trait.poly<0>], @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] { func.func private @b() -> i64 }
trait.impl private @A_i32 for @A[i32] {
  func.func @a() -> i64 {
    %c = arith.constant 32 : i64
    return %c : i64
  }
}
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  func.func @b() -> i64 {
    %s = trait.assume self : !trait.claim<@B[!trait.poly<0>]>
    %a = trait.project %s[1] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    return %r : i64
  }
}
// expected-error @below {{proof @A_i32 proves '!trait.claim<@A[i32]>', which does not discharge the obligation '!trait.claim<@A[i64]>'}}
trait.proof private @forged proves @B_blanket[!trait.poly<0> = i32] for @B[i32] given [@Foo_i32, @A_i32]
