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
// VERIFIED: trait.derive @B[tuple<i32>] from @B_tuple[i32] given(%[[A]], %{{.*}})
// VERIFIED: trait.project %{{.*}}[2] : <@B[tuple<i32>] by @p> -> <@A[i32] by @A_i32>

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A(!trait.claim<@A[!S]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!S]>, !S) -> i64
trait.trait private @C(!trait.claim<@C[!S]>) assoc {Val = 0}
trait.trait private @B(!trait.claim<@B[!S]>) -> (!trait.claim<@A[!S]>, !trait.claim<!trait.proj<@B[!S], "Out"> = i64>) assoc {Out = 0} members [@B.b]
func.func private @B.b(!trait.claim<@B[!S]>, !S) -> i64

trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>, %x: i32) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}
trait.impl private @A_tuple(%self: !trait.claim<@A[tuple<!trait.poly<0>>]>) members [@A.a -> @A_tuple.a] {
}
func.func private @A_tuple.a(%self: !trait.claim<@A[tuple<!trait.poly<0>>]>, %x: tuple<!trait.poly<0>>) -> i64 {
  %c = arith.constant 11 : i64
  return %c : i64
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) bindings {Val = i64} {
}
trait.impl private @B_tuple(%self: !trait.claim<@B[tuple<!trait.poly<0>>]>, %a: !trait.claim<@A[!trait.poly<0>]>, %val: !trait.claim<!trait.proj<@C[!trait.poly<0>], "Val"> = i64>) bindings {Out = i64} members [@B.b -> @B_tuple.b] {
  %req0 = trait.allege @A[tuple<!trait.poly<0>>]
  %out = trait.witness proj_resolve !trait.proj<@B[tuple<!trait.poly<0>>], "Out"> resolves i64 by @B_tuple[!trait.poly<0>]
    given(%a, %val) : (!trait.claim<@A[!trait.poly<0>]>, !trait.claim<!trait.proj<@C[!trait.poly<0>], "Val"> = i64>)
    : !trait.claim<!trait.proj<@B[tuple<!trait.poly<0>>], "Out"> = i64>
  trait.return %req0, %out : !trait.claim<@A[tuple<!trait.poly<0>>]>, !trait.claim<!trait.proj<@B[tuple<!trait.poly<0>>], "Out"> = i64>
}
func.func private @B_tuple.b(%self: !trait.claim<@B[tuple<!trait.poly<0>>]>, %a: !trait.claim<@A[!trait.poly<0>]>, %val: !trait.claim<!trait.proj<@C[!trait.poly<0>], "Val"> = i64>, %x: tuple<!trait.poly<0>>) -> i64 {
  %c = arith.constant 35 : i64
  return %c : i64
}
trait.proof private @p {
  %p0 = trait.witness @A_i32 for @A[i32]
  %p1 = trait.witness proj_resolve !trait.proj<@C[i32], "Val"> resolves i64 by @C_i32
    : !trait.claim<!trait.proj<@C[i32], "Val"> = i64>
  %d = trait.derive @B[tuple<i32>] from @B_tuple[i32] given(%p0, %p1) : (!trait.claim<@A[i32] by @A_i32>, !trait.claim<!trait.proj<@C[i32], "Val"> = i64>)
  trait.return %d : !trait.claim<@B[tuple<i32>]>
}

func.func @main(%x: tuple<i32>, %y: i32) -> i64 {
  %w = trait.witness @p for @B[tuple<i32>]
  %r = trait.func.call @B.b(%w, %x) : (!trait.claim<@B[tuple<i32>] by @p>, tuple<i32>) -> i64
  %a = trait.project %w[2] : !trait.claim<@B[tuple<i32>] by @p> -> !trait.claim<@A[i32] by @A_i32>
  %s = trait.func.call @A.a(%a, %y) : (!trait.claim<@A[i32] by @A_i32>, i32) -> i64
  %t = arith.addi %r, %s : i64
  return %t : i64
}

// CHECK-DAG: func.func private @[[B:B_tuple.b_h[0-9a-f]+]](%{{.*}}: tuple<i32>) -> i64
// CHECK-DAG: func.func private @[[A:A_i32.a_h[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: func.func @main
// CHECK-DAG: call @[[B]](
// CHECK-DAG: call @[[A]](
// CHECK-NOT: trait.
