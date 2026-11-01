// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait's default reads its trait's self claim, an application requirement
// and an equality requirement off the self claim by position. Cut for an impl
// that leaves the member to the default, each requirement is the evidence the
// impl returns at that position; the instance then lowers with the
// requirement's member called and the equality discharged.

// VERIFIED: func.func private @A.twice(%[[SELF:[^:]*]]: !trait.claim<@A[!trait.poly<0>]>
// VERIFIED: trait.project %[[SELF]][1] : <@A[!trait.poly<0>]> -> <!trait.proj<@A[!trait.poly<0>], "Out"> = i64>

!S = !trait.poly<0>

trait.trait private @B(!trait.claim<@B[!S]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!S]>, !S) -> i64

trait.trait private @A(!trait.claim<@A[!S]>) -> (!trait.claim<@B[!S]>, !trait.claim<!trait.proj<@A[!S], "Out"> = i64>) assoc {Out = 0} members [@A.make, default @A.twice]
func.func private @A.make(!trait.claim<@A[!S]>, !S) -> !trait.proj<@A[!S], "Out">
func.func private @A.twice(%self: !trait.claim<@A[!S]>, %x: !S) -> i64 {
  %b = trait.project %self[0] : !trait.claim<@A[!S]> -> !trait.claim<@B[!S]>
  %e = trait.project %self[1] : !trait.claim<@A[!S]> -> !trait.claim<!trait.proj<@A[!S], "Out"> = i64>
  %o = trait.func.call @A.make(%self, %x) : (!trait.claim<@A[!S]>, !S) -> !trait.proj<@A[!S], "Out">
  %oi = trait.coerce %o : !trait.proj<@A[!S], "Out"> to i64 via (%e) : (!trait.claim<!trait.proj<@A[!S], "Out"> = i64>)
  %v = trait.func.call @B.b(%b, %x) : (!trait.claim<@B[!S]>, !S) -> i64
  %r = arith.addi %oi, %v : i64
  return %r : i64
}

trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) bindings {Out = i64} members [@A.make -> @A_i32.make] {
  %b = trait.witness @B_i32 for @B[i32]
  %out = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i64 by @A_i32 : !trait.claim<!trait.proj<@A[i32], "Out"> = i64>
  trait.return %b, %out : !trait.claim<@B[i32] by @B_i32>, !trait.claim<!trait.proj<@A[i32], "Out"> = i64>
}
func.func private @A_i32.make(%self: !trait.claim<@A[i32]>, %x: i32) -> i64 {
  %c = arith.constant 35 : i64
  return %c : i64
}

func.func @main(%x: i32) -> i64 {
  %a = trait.allege @A[i32]
  %r = trait.func.call @A.twice(%a, %x) : (!trait.claim<@A[i32]>, i32) -> i64
  return %r : i64
}

// CHECK: func.func private @[[TWICE:A.twice_h[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK-NEXT: call @A_i32.make_h{{[0-9a-f]+}}(
// CHECK-NEXT: call @B_i32.b_h{{[0-9a-f]+}}(
// CHECK-DAG: func.func private @B_i32.b_h{{[0-9a-f]+}}(%{{.*}}: i32) -> i64
// CHECK-DAG: func.func private @A_i32.make_h{{[0-9a-f]+}}(%{{.*}}: i32) -> i64
// CHECK: func.func @main
// CHECK: call @[[TWICE]](
// CHECK-NOT: trait.
