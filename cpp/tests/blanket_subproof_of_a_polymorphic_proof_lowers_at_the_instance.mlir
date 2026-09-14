// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @p stands over two variables and cites @a2, a proof standing over two
// variables of its own. The claim a subproof stands over is the obligation it
// discharges, spelled in the variables @p stands over, so cloning @p at
// @B2[i32, i32] respells it. Spelled in @a2's own variables instead, nothing
// the clone substitutes would reach them and the monomorphic body would keep
// them.

trait.trait private @A2[!trait.poly<0>, !trait.poly<1>] { func.func private @a() -> i64 }
trait.trait private @B2[!trait.poly<0>, !trait.poly<1>] where [@A2[!trait.poly<0>, !trait.poly<1>]] { func.func private @b() -> i64 }
trait.impl private @A2_blanket for @A2[!trait.poly<2>, !trait.poly<3>] {
  func.func @a() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @B2_blanket for @B2[!trait.poly<0>, !trait.poly<1>] {
  func.func @b() -> i64 {
    %s = trait.assume @B2[!trait.poly<0>, !trait.poly<1>]
    %a = trait.project %s[0] : !trait.claim<@B2[!trait.poly<0>, !trait.poly<1>]> -> !trait.claim<@A2[!trait.poly<0>, !trait.poly<1>]>
    %r = trait.method.call %a @A2[!trait.poly<0>, !trait.poly<1>]::@a() : () -> i64
    return %r : i64
  }
}
trait.proof private @a2 proves @A2_blanket for @A2[!trait.poly<2>, !trait.poly<3>] given []
trait.proof private @p proves @B2_blanket for @B2[!trait.poly<0>, !trait.poly<1>] given [@a2]

// CHECK-NOT: trait.
// CHECK: func.func private @[[A:A2_blanket_[a-z0-9]+]]_a() -> i64
// CHECK: func.func private @[[B:B2_blanket_[a-z0-9]+]]_b() -> i64
// CHECK: call @[[A]]_a() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @[[B]]_b() : () -> i64
// CHECK-NOT: trait.
func.func @main() -> i64 {
  %w = trait.witness @p for @B2[i32, i32]
  %r = trait.method.call %w @B2[i32, i32]::@b() : () -> i64 by @p
  return %r : i64
}
