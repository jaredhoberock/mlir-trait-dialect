// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @p stands over poly<7>, poly<8> for an impl written over poly<0>, poly<1>,
// and cites @a2, which stands over poly<4>, poly<5> for an impl written over
// poly<2>, poly<3>. The claim a subproof stands over is the obligation it
// discharges at the application the citation names, so witnessing @p at
// @B2[i32, i32] spells its subproof's claim there. Spelled at @p's own
// declaration instead, the clone of @B2_blanket at i32 would keep a claim over
// poly<7>, poly<8> that no substitution reaches.

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
trait.proof private @a2 proves @A2_blanket for @A2[!trait.poly<4>, !trait.poly<5>] given []
trait.proof private @p proves @B2_blanket for @B2[!trait.poly<7>, !trait.poly<8>] given [@a2]

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
