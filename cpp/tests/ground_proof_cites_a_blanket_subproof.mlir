// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A ground proof of @B2[i32, i32] discharges its obligation @A2[i32, i32] with
// @a2, which stands over every instance of @A2. The obligation is what the
// subproof stands over, so the body @B2_i32 already spells at i32 finds the
// evidence spelled there too.

trait.trait private @A2[!trait.poly<0>, !trait.poly<1>] { func.func private @a() -> i64 }
trait.trait private @B2[!trait.poly<0>, !trait.poly<1>] where [@A2[!trait.poly<0>, !trait.poly<1>]] { func.func private @b() -> i64 }
trait.impl private @A2_blanket for @A2[!trait.poly<2>, !trait.poly<3>] {
  func.func @a() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @B2_i32 for @B2[i32, i32] {
  func.func @b() -> i64 {
    %s = trait.assume @B2[i32, i32]
    %a = trait.project %s[0] : !trait.claim<@B2[i32, i32]> -> !trait.claim<@A2[i32, i32]>
    %r = trait.method.call %a @A2[i32, i32]::@a() : () -> i64
    return %r : i64
  }
}
trait.proof private @a2 proves @A2_blanket for @A2[!trait.poly<2>, !trait.poly<3>] given []
trait.proof private @pb proves @B2_i32 for @B2[i32, i32] given [@a2]

// CHECK-NOT: trait.
// CHECK: func.func private @[[A:A2_blanket_[a-z0-9]+]]_a() -> i64
// CHECK: func.func private @B2_i32_b() -> i64
// CHECK: call @[[A]]_a() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @B2_i32_b() : () -> i64
// CHECK-NOT: trait.
func.func @main() -> i64 {
  %w = trait.witness @pb for @B2[i32, i32]
  %r = trait.method.call %w @B2[i32, i32]::@b() : () -> i64 by @pb
  return %r : i64
}
