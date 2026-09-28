// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A method cites its declaration's self application and where-clause entries
// by position, each spelling the claim the entry states. The impl's first and
// third entries spell one claim and stay two citations.

!T = !trait.poly<0>

trait.trait private @B[!T] {
  trait.assoc_type @Out
  func.func private @b(!T) -> i64
}

trait.trait private @A[!T] where [@B[!T]] {
  func.func private @a(!T) -> i64
  func.func @twice(%x: !T) -> i64 {
    %s = trait.assume self : !trait.claim<@A[!T]>
    %b = trait.assume 0 : !trait.claim<@B[!T]>
    %u = trait.method.call %s @A[!T]::@a(%x) : (!T) -> i64
    %v = trait.method.call %b @B[!T]::@b(%x) : (!T) -> i64
    %r = arith.addi %u, %v : i64
    return %r : i64
  }
}

trait.impl private @A_gen for @A[!T] where [@B[!T], !trait.proj<@B[!T], "Out"> = i64, @B[!T]] {
  func.func @a(%x: !T) -> i64 {
    %s = trait.assume self : !trait.claim<@A[!T]>
    %b0 = trait.assume 0 : !trait.claim<@B[!T]>
    %e = trait.assume 1 : !trait.claim<!trait.proj<@B[!T], "Out"> = i64>
    %b2 = trait.assume 2 : !trait.claim<@B[!T]>
    %r = trait.method.call %b2 @B[!T]::@b(%x) : (!T) -> i64
    return %r : i64
  }
}

// CHECK: trait.trait private @A
// CHECK: trait.assume self : !trait.claim<@A[!trait.poly<0>]>
// CHECK: trait.assume 0 : !trait.claim<@B[!trait.poly<0>]>
// CHECK: trait.impl private @A_gen
// CHECK: trait.assume self : !trait.claim<@A[!trait.poly<0>]>
// CHECK: trait.assume 0 : !trait.claim<@B[!trait.poly<0>]>
// CHECK: trait.assume 1 : !trait.claim<!trait.proj<@B[!trait.poly<0>], "Out"> = i64>
// CHECK: trait.assume 2 : !trait.claim<@B[!trait.poly<0>]>
