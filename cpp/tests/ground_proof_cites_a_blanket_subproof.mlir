// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A ground proof of @B2[i32, i32] stands beside @a2, a proof of @A2[i32, i32]
// by the blanket impl. The requirement @B2_i32 alleges is spelled at i32, so
// the body @B2_i32 already spells at i32 finds the evidence spelled there too.

trait.trait private @A2(%self: !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>) { trait.method @a() -> i64 }
trait.trait private @B2(%self: !trait.claim<@B2[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]> { trait.method @b() -> i64 }
trait.impl private @A2_blanket(%self: !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.impl private @B2_i32(%self: !trait.claim<@B2[i32, i32]>) {
  trait.method @b() -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B2[i32, i32]> -> !trait.claim<@A2[i32, i32]>
    %r = trait.method.call %a @A2[i32, i32]::@a() : () -> i64
    trait.return %r : i64
  }
  %req0 = trait.allege @A2[i32, i32]
  trait.return %req0 : !trait.claim<@A2[i32, i32]>
}
trait.proof private @a2 {
  %d = trait.derive @A2[i32, i32] from @A2_blanket[i32, i32] given()
  trait.return %d : !trait.claim<@A2[i32, i32]>
}
trait.proof private @pb {
  %d = trait.derive @B2[i32, i32] from @B2_i32 given()
  trait.return %d : !trait.claim<@B2[i32, i32]>
}

// CHECK-NOT: trait.
// CHECK: func.func private @[[A:A2_blanket_[a-z0-9]+]]_a() -> i64
// CHECK: func.func private @B2_i32_{{h[0-9a-f]+}}_b() -> i64
// CHECK: call @[[A]]_a() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @B2_i32_{{h[0-9a-f]+}}_b() : () -> i64
// CHECK-NOT: trait.
func.func @main() -> i64 {
  %w = trait.witness @pb for @B2[i32, i32]
  %r = trait.method.call %w @B2[i32, i32]::@b() : () -> i64 by @pb
  return %r : i64
}
