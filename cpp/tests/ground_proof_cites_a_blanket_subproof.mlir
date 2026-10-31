// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A ground proof of @B2[i32, i32] stands beside @a2, a proof of @A2[i32, i32]
// by the blanket impl. The requirement @B2_i32 alleges is spelled at i32, so
// the body @B2_i32 already spells at i32 finds the evidence spelled there too.

trait.trait private @A2(!trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>) members [@A2.a]
func.func private @A2.a(!trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>) -> i64
trait.trait private @B2(!trait.claim<@B2[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]> members [@B2.b]
func.func private @B2.b(!trait.claim<@B2[!trait.poly<0>, !trait.poly<1>]>) -> i64
trait.impl private @A2_blanket(%self: !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>) members [@A2.a -> @A2_blanket.a] {
}
func.func private @A2_blanket.a(%self: !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}
trait.impl private @B2_i32(%self: !trait.claim<@B2[i32, i32]>) members [@B2.b -> @B2_i32.b] {
  %req0 = trait.allege @A2[i32, i32]
  trait.return %req0 : !trait.claim<@A2[i32, i32]>
}
func.func private @B2_i32.b(%self: !trait.claim<@B2[i32, i32]>) -> i64 {
  %a = trait.project %self[0] : !trait.claim<@B2[i32, i32]> -> !trait.claim<@A2[i32, i32]>
  %r = trait.func.call @A2.a(%a) : (!trait.claim<@A2[i32, i32]>) -> i64
  return %r : i64
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
// CHECK: func.func private @[[A:A2_blanket.a_[a-z0-9]+]]() -> i64
// CHECK: func.func private @B2_i32.b_{{h[0-9a-f]+}}() -> i64
// CHECK: call @[[A]]() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @B2_i32.b_{{h[0-9a-f]+}}() : () -> i64
// CHECK-NOT: trait.
func.func @main() -> i64 {
  %w = trait.witness @pb for @B2[i32, i32]
  %r = trait.func.call @B2.b(%w) : (!trait.claim<@B2[i32, i32] by @pb>) -> i64
  return %r : i64
}
