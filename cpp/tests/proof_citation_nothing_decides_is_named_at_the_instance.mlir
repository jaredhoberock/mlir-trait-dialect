// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --check-prefix=INSTANCE

// @B's requirement projects through @Foo, and no impl of @Foo exists.
// @B_blanket alleges the requirement, which the impl's return check accepts and
// the module verifies. At the instance the hop off the proven claim is replaced
// by that allegation, selection cannot prove it either, and the refusal names
// the allegation where the impl wrote it, called from the hop in the instance.

// VERIFIED: trait.proof private @forged {
// VERIFIED: trait.derive @B[!trait.poly<0>] from @B_blanket
// INSTANCE: :33:{{[0-9]+}}: error: unproven monomorphic claim '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>' after instantiate-monomorphs
// INSTANCE: :29:{{[0-9]+}}: note: called from

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> { trait.method @b() -> i64 }
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @b() -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
  %req0 = trait.allege @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]
  trait.return %req0 : !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
}
trait.proof private @forged {
  %d = trait.derive @B[!trait.poly<0>] from @B_blanket given()
  trait.return %d : !trait.claim<@B[!trait.poly<0>]>
}
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
