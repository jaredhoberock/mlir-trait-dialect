// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A conditional impl's method cites its where clause's application entry and
// equality entry by position. Extracted as a free function for the instance a
// call wants, each citation becomes the requirement at that position of the
// leading proof of the impl's self application, and the instance lowers with
// the entry's method called and the equality discharged.

!T = !trait.poly<0>

trait.trait private @B[!T] {
  trait.assoc_type @Out
  func.func private @b(!T) -> !trait.proj<@B[!T], "Out">
}

trait.trait private @A[!T] {
  func.func private @a(!T) -> i64
}

trait.impl private @B_i32 for @B[i32] {
  trait.assoc_type @Out = i64
  func.func @b(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

trait.impl private @A_gen for @A[!T] where [@B[!T], !trait.proj<@B[!T], "Out"> = i64] {
  func.func @a(%x: !T) -> i64 {
    %b = trait.assume 0 : !trait.claim<@B[!T]>
    %e = trait.assume 1 : !trait.claim<!trait.proj<@B[!T], "Out"> = i64>
    %o = trait.method.call %b @B[!T]::@b(%x) : (!T) -> !trait.proj<@B[!T], "Out">
    %r = trait.coerce %o : !trait.proj<@B[!T], "Out"> to i64 via (%e) : (!trait.claim<!trait.proj<@B[!T], "Out"> = i64>)
    return %r : i64
  }
}

func.func @main(%x: i32) -> i64 {
  %a = trait.allege @A[i32]
  %r = trait.method.call %a @A[i32]::@a(%x) : (i32) -> i64
  return %r : i64
}

// CHECK: func.func private @[[B:B_i32_b[_a-z0-9]*]](%{{.*}}: i32) -> i64
// CHECK: func.func private @[[A:A_gen_[_a-z0-9]*]](%{{.*}}: i32) -> i64
// CHECK-NEXT: call @[[B]](
// CHECK: func.func @main
// CHECK: call @[[A]](
// CHECK-NOT: trait.
