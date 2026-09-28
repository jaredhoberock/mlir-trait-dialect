// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A proof states the argument its impl's parameter takes, and its given list
// holds one entry per requirement of the trait and per entry of the impl's
// where clause: a symbol for each application, unit for each equality,
// decided at the proof's claim. A requirement is read off the proof at its own
// position, carrying the symbol standing there.

// VERIFIED: trait.proof private @p proves @B_tuple[!trait.poly<1> = i32] for @B[tuple<i32>] given [@A_tuple_p, unit, @A_i32, unit]
// VERIFIED: trait.project %{{.*}}[2] : <@B[tuple<i32>] by @p> -> <@A[i32] by @A_i32>

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A[!S] {
  func.func private @a(!S) -> i64
}
trait.trait private @C[!S] {
  trait.assoc_type @Val
}
trait.trait private @B[!S] where [@A[!S], !trait.proj<@B[!S], "Out"> = i64] {
  trait.assoc_type @Out
  func.func private @b(!S) -> i64
}

trait.impl private @A_i32 for @A[i32] {
  func.func @a(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @A_tuple for @A[tuple<!U>] {
  func.func @a(%x: tuple<!U>) -> i64 {
    %c = arith.constant 11 : i64
    return %c : i64
  }
}
trait.proof private @A_tuple_p proves @A_tuple for @A[tuple<i32>] given []
trait.impl private @C_i32 for @C[i32] {
  trait.assoc_type @Val = i64
}
trait.impl private @B_tuple for @B[tuple<!U>] where [@A[!U], !trait.proj<@C[!U], "Val"> = i64] {
  trait.assoc_type @Out = i64
  func.func @b(%x: tuple<!U>) -> i64 {
    %c = arith.constant 35 : i64
    return %c : i64
  }
}
trait.proof private @p proves @B_tuple[!U = i32] for @B[tuple<i32>] given [@A_tuple_p, unit, @A_i32, unit]

func.func @main(%x: tuple<i32>, %y: i32) -> i64 {
  %w = trait.witness @p for @B[tuple<i32>]
  %r = trait.method.call %w @B[tuple<i32>]::@b(%x) : (tuple<i32>) -> i64 by @p
  %a = trait.project %w[2] : !trait.claim<@B[tuple<i32>] by @p> -> !trait.claim<@A[i32] by @A_i32>
  %s = trait.method.call %a @A[i32]::@a(%y) : (i32) -> i64 by @A_i32
  %t = arith.addi %r, %s : i64
  return %t : i64
}

// CHECK-DAG: func.func private @[[B:B_tuple_[_a-z0-9]*b[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK-DAG: func.func private @[[A:A_i32_a]](%{{.*}}: i32) -> i64
// CHECK: func.func @main
// CHECK-DAG: call @[[B]](
// CHECK-DAG: call @[[A]](
// CHECK-NOT: trait.
