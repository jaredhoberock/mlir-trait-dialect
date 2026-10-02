// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait default method reads its trait's self claim, an application
// requirement and an equality requirement off the self claim by position. Cut
// for an impl that does not define the method, each requirement is the
// evidence the impl returns at that position; the instance then lowers with
// the requirement's method called and the equality discharged.

// VERIFIED: trait.project %self[1] : <@A[!trait.poly<0>]> -> <!trait.proj<@A[!trait.poly<0>], "Out"> = i64>

!S = !trait.poly<0>

trait.trait private @B(%self: !trait.claim<@B[!S]>) {
  trait.method @b(!S) -> i64
}

trait.trait private @A(%self: !trait.claim<@A[!S]>) -> (!trait.claim<@B[!S]>, !trait.claim<!trait.proj<@A[!S], "Out"> = i64>) {
  trait.assoc_type @Out
  trait.method @make(!S) -> !trait.proj<@A[!S], "Out">
  trait.method @twice(%x: !S) -> i64 {
    %b = trait.project %self[0] : !trait.claim<@A[!S]> -> !trait.claim<@B[!S]>
    %e = trait.project %self[1] : !trait.claim<@A[!S]> -> !trait.claim<!trait.proj<@A[!S], "Out"> = i64>
    %o = trait.method.call %self @A[!S]::@make(%x) : (!S) -> !trait.proj<@A[!S], "Out">
    %oi = trait.coerce %o : !trait.proj<@A[!S], "Out"> to i64 via (%e) : (!trait.claim<!trait.proj<@A[!S], "Out"> = i64>)
    %v = trait.method.call %b @B[!S]::@b(%x) : (!S) -> i64
    %r = arith.addi %oi, %v : i64
    trait.return %r : i64
  }
}

trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @b(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}

trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.assoc_type @Out = i64
  trait.method @make(%x: i32) -> i64 {
    %c = arith.constant 35 : i64
    trait.return %c : i64
  }
  %b = trait.witness @B_i32 for @B[i32]
  %out = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i64 by @A_i32 : !trait.claim<!trait.proj<@A[i32], "Out"> = i64>
  trait.return %b, %out : !trait.claim<@B[i32] by @B_i32>, !trait.claim<!trait.proj<@A[i32], "Out"> = i64>
}

func.func @main(%x: i32) -> i64 {
  %a = trait.allege @A[i32]
  %r = trait.method.call %a @A[i32]::@twice(%x) : (i32) -> i64
  return %r : i64
}

// CHECK: func.func private @[[TWICE:A_h[0-9a-f]+_twice]](%{{.*}}: i32) -> i64
// CHECK-NEXT: call @A_i32_h{{[0-9a-f]+}}_make(
// CHECK-NEXT: call @B_i32_h{{[0-9a-f]+}}_b(
// CHECK-DAG: func.func private @B_i32_h{{[0-9a-f]+}}_b(%{{.*}}: i32) -> i64
// CHECK-DAG: func.func private @A_i32_h{{[0-9a-f]+}}_make(%{{.*}}: i32) -> i64
// CHECK: func.func @main
// CHECK: call @[[TWICE]](
// CHECK-NOT: trait.
