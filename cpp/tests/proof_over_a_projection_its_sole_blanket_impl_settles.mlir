// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @B's requirement projects through @Foo, whose one impl carries no premise and
// binds Out = i64 for every argument. That impl serves @P's claim, so the
// obligation @B_blanket's requirement states at @P is @A[i64], and the
// requirement it alleges is proved there by @A_i64. A reading that left the
// projection standing would leave a valid proof with an obligation nothing
// discharges.

// CHECK-NOT: trait.
// CHECK: func.func private @[[A:A_i64_h[0-9a-f]+_a]]() -> i64
// CHECK: func.func private @[[B:B_blanket_[a-z0-9]+]]_b() -> i64
// CHECK: call @[[A]]() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @[[B]]_b() : () -> i64
// CHECK-NOT: trait.

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out = i64 }
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
trait.proof private @P {
  %d = trait.derive @B[i32] from @B_blanket[i32] given()
  trait.return %d : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @P for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @P
  return %r : i64
}
