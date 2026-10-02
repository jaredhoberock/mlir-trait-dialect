// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @B's requirement projects through @Foo, whose one impl carries no premise and
// binds Out = i64 for every argument. That impl serves every instance of @P's
// variable, so the obligation @P states is @A[i64] wherever @P stands, and the
// citation of @A_i64 discharges it at @P's own claim -- not only at the
// instances a witness names. A reading that left the projection standing would
// leave a valid proof with an obligation nothing discharges.

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
  %d = trait.derive @B[!trait.poly<0>] from @B_blanket given()
  trait.return %d : !trait.claim<@B[!trait.poly<0>]>
}
func.func @main() -> i64 {
  %w = trait.witness @P for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @P
  return %r : i64
}
