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
// CHECK: func.func private @[[A:A_i64_a]]() -> i64
// CHECK: func.func private @[[B:B_blanket_[a-z0-9]+]]_b() -> i64
// CHECK: call @[[A]]() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @[[B]]_b() : () -> i64
// CHECK-NOT: trait.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = i64 }
trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] { func.func private @b() -> i64 }
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
trait.proof private @P proves @B_blanket for @B[!trait.poly<0>] given [@A_i64]
func.func @main() -> i64 {
  %w = trait.witness @P for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @P
  return %r : i64
}
