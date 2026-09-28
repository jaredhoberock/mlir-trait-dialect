// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait default method cites its trait's self application, an application
// requirement and an equality requirement by position. Cloned into an impl
// that does not define the method, a where-clause position would name the
// impl's own entries, so the clone selects each requirement off the impl's
// self claim; the instance then lowers with the requirement's method called
// and the equality discharged.

// VERIFIED: trait.assume 1 : !trait.claim<!trait.proj<@A[!trait.poly<0>], "Out"> = i64>

!S = !trait.poly<0>

trait.trait private @B[!S] {
  func.func private @b(!S) -> i64
}

trait.trait private @A[!S] where [@B[!S], !trait.proj<@A[!S], "Out"> = i64] {
  trait.assoc_type @Out
  func.func private @make(!S) -> !trait.proj<@A[!S], "Out">
  func.func @twice(%x: !S) -> i64 {
    %s = trait.assume self : !trait.claim<@A[!S]>
    %b = trait.assume 0 : !trait.claim<@B[!S]>
    %e = trait.assume 1 : !trait.claim<!trait.proj<@A[!S], "Out"> = i64>
    %o = trait.method.call %s @A[!S]::@make(%x) : (!S) -> !trait.proj<@A[!S], "Out">
    %oi = trait.coerce %o : !trait.proj<@A[!S], "Out"> to i64 via (%e) : (!trait.claim<!trait.proj<@A[!S], "Out"> = i64>)
    %v = trait.method.call %b @B[!S]::@b(%x) : (!S) -> i64
    %r = arith.addi %oi, %v : i64
    return %r : i64
  }
}

trait.impl private @B_i32 for @B[i32] {
  func.func @b(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

trait.impl private @A_i32 for @A[i32] {
  trait.assoc_type @Out = i64
  func.func @make(%x: i32) -> i64 {
    %c = arith.constant 35 : i64
    return %c : i64
  }
}

func.func @main(%x: i32) -> i64 {
  %a = trait.allege @A[i32]
  %r = trait.method.call %a @A[i32]::@twice(%x) : (i32) -> i64
  return %r : i64
}

// CHECK-DAG: func.func private @[[MAKE:A_i32_make[_a-z0-9]*]](%{{.*}}: i32) -> i64
// CHECK-DAG: func.func private @[[B:B_i32_b[_a-z0-9]*]](%{{.*}}: i32) -> i64
// CHECK: func.func private @[[TWICE:A_i32_twice[_a-z0-9]*]](%{{.*}}: i32) -> i64
// CHECK: call @[[MAKE]](
// CHECK: call @[[B]](
// CHECK: func.func @main
// CHECK: call @[[TWICE]](
// CHECK-NOT: trait.
