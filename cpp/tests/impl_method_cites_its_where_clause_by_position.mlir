// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A conditional impl's member function cites its where clause's application
// entry and equality entry by position, its leading parameters there. A call
// of the member through a proven receiver passes the premises the receiver's
// proof derives the impl from at those positions, and the instance lowers with
// the entry's member called and the equality discharged.

!T = !trait.poly<0>

trait.trait private @B(!trait.claim<@B[!T]>) assoc {Out = 0} members [@B.b]
func.func private @B.b(!trait.claim<@B[!T]>, !T) -> !trait.proj<@B[!T], "Out">

trait.trait private @A(!trait.claim<@A[!T]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!T]>, !T) -> i64

trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) bindings {Out = i64} members [@B.b -> @B_i32.b] {
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b_1: !trait.claim<@B[!T]>, %out: !trait.claim<!trait.proj<@B[!T], "Out"> = i64>) members [@A.a -> @A_gen.a] {
}
func.func private @A_gen.a(%self: !trait.claim<@A[!T]>, %b_1: !trait.claim<@B[!T]>, %out: !trait.claim<!trait.proj<@B[!T], "Out"> = i64>, %x: !T) -> i64 {
  %o = trait.func.call @B.b(%b_1, %x) : (!trait.claim<@B[!T]>, !T) -> !trait.proj<@B[!T], "Out">
  %r = trait.coerce %o : !trait.proj<@B[!T], "Out"> to i64 via (%out) : (!trait.claim<!trait.proj<@B[!T], "Out"> = i64>)
  return %r : i64
}

func.func @main(%x: i32) -> i64 {
  %a = trait.allege @A[i32]
  %r = trait.func.call @A.a(%a, %x) : (!trait.claim<@A[i32]>, i32) -> i64
  return %r : i64
}

// CHECK: func.func private @[[B:B_i32.b_h[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: func.func private @[[A:A_gen.a_h[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK-NEXT: call @[[B]](
// CHECK: func.func @main
// CHECK: call @[[A]](
// CHECK-NOT: trait.
