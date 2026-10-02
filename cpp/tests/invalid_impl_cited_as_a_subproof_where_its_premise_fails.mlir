// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @I applies where i32 is i64, which it is not. @Uses_i32 returns, for its
// trait's requirement @T[i32], a derive of @I over an alleged i32 = i64. A
// derive is proven by impl selection where it is used, and selection finds no
// impl of @T[i32] whose premises hold, so the use is refused rather than run
// through @I.

// CHECK: error: unproven monomorphic claim '!trait.claim<@T[i32]>' after instantiate-monomorphs

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {
  trait.method @m() -> i64
}
trait.impl private @I(%self: !trait.claim<@T[i32]>, %eq: !trait.claim<i32 = i64>) {
  trait.method @m() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}
trait.trait private @Uses(%self: !trait.claim<@Uses[!trait.poly<0>]>) -> !trait.claim<@T[!trait.poly<0>]> {
  trait.method @u() -> i64
}
trait.impl private @Uses_i32(%self: !trait.claim<@Uses[i32]>) {
  %eq = trait.allege i32 = i64
  %t = trait.derive @T[i32] from @I given(%eq) : (!trait.claim<i32 = i64>)
  trait.method @u() -> i64 {
    %c = arith.constant 3 : i64
    trait.return %c : i64
  }
  trait.return %t : !trait.claim<@T[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @Uses_i32 for @Uses[i32]
  %t = trait.project %w[0] : !trait.claim<@Uses[i32] by @Uses_i32> -> !trait.claim<@T[i32]>
  %r = trait.method.call %t @T[i32]::@m() : () -> i64
  return %r : i64
}
