// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A proof's derive supplies one claim per entry of the impl's where clause,
// in order, and those claims together with the derived application determine
// the argument the impl's parameter takes; nothing states it a second time.
// The trait's requirements are the impl's to return. A projection off the proof
// reads requirement k from the impl's return and, past the requirements, where
// entry k - 2 from the derive's operand, carrying the symbol standing there.

// VERIFIED: trait.proof private @p {
// VERIFIED: %[[A:.*]] = trait.witness @A_i32 for @A[i32]
// VERIFIED: trait.derive @B[tuple<i32>] from @B_tuple given(%[[A]], %{{.*}})
// VERIFIED: trait.project %{{.*}}[2] : <@B[tuple<i32>] by @p> -> <@A[i32] by @A_i32>

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A(%self: !trait.claim<@A[!S]>) {
  trait.method @a(!S) -> i64
}
trait.trait private @C(%self: !trait.claim<@C[!S]>) {
  trait.assoc_type @Val
}
trait.trait private @B(%self: !trait.claim<@B[!S]>) -> (!trait.claim<@A[!S]>, !trait.claim<!trait.proj<@B[!S], "Out"> = i64>) {
  trait.assoc_type @Out
  trait.method @b(!S) -> i64
}

trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @a(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_tuple(%self: !trait.claim<@A[tuple<!U>]>) {
  trait.method @a(%x: tuple<!U>) -> i64 {
    %c = arith.constant 11 : i64
    trait.return %c : i64
  }
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) {
  trait.assoc_type @Val = i64
}
trait.impl private @B_tuple(%self: !trait.claim<@B[tuple<!U>]>, %a: !trait.claim<@A[!U]>, %val: !trait.claim<!trait.proj<@C[!U], "Val"> = i64>) {
  trait.assoc_type @Out = i64
  trait.method @b(%x: tuple<!U>) -> i64 {
    %c = arith.constant 35 : i64
    trait.return %c : i64
  }
  %req0 = trait.allege @A[tuple<!U>]
  %out = trait.witness proj_resolve !trait.proj<@B[tuple<!U>], "Out"> resolves i64 by @B_tuple
    given(%a, %val) : (!trait.claim<@A[!U]>, !trait.claim<!trait.proj<@C[!U], "Val"> = i64>)
    : !trait.claim<!trait.proj<@B[tuple<!U>], "Out"> = i64>
  trait.return %req0, %out : !trait.claim<@A[tuple<!U>]>, !trait.claim<!trait.proj<@B[tuple<!U>], "Out"> = i64>
}
trait.proof private @p {
  %p0 = trait.witness @A_i32 for @A[i32]
  %p1 = trait.witness proj_resolve !trait.proj<@C[i32], "Val"> resolves i64 by @C_i32
    : !trait.claim<!trait.proj<@C[i32], "Val"> = i64>
  %d = trait.derive @B[tuple<i32>] from @B_tuple given(%p0, %p1) : (!trait.claim<@A[i32] by @A_i32>, !trait.claim<!trait.proj<@C[i32], "Val"> = i64>)
  trait.return %d : !trait.claim<@B[tuple<i32>]>
}

func.func @main(%x: tuple<i32>, %y: i32) -> i64 {
  %w = trait.witness @p for @B[tuple<i32>]
  %r = trait.method.call %w @B[tuple<i32>]::@b(%x) : (tuple<i32>) -> i64 by @p
  %a = trait.project %w[2] : !trait.claim<@B[tuple<i32>] by @p> -> !trait.claim<@A[i32] by @A_i32>
  %s = trait.method.call %a @A[i32]::@a(%y) : (i32) -> i64 by @A_i32
  %t = arith.addi %r, %s : i64
  return %t : i64
}

// CHECK-DAG: func.func private @[[B:B_tuple_[_a-z0-9]*b[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK-DAG: func.func private @[[A:A_i32_h[0-9a-f]+_a]](%{{.*}}: i32) -> i64
// CHECK: func.func @main
// CHECK-DAG: call @[[B]](
// CHECK-DAG: call @[[A]](
// CHECK-NOT: trait.
