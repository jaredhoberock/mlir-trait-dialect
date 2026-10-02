// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A method cites its declaration's block arguments -- the self claim, then one
// claim per where entry -- by position, each spelling the claim the entry
// states; a trait method reads a requirement off its self claim by index. The
// impl's first and third entries spell one claim and stay two arguments: its
// method reads the third and its return the first.

!T = !trait.poly<0>

trait.trait private @B(%self: !trait.claim<@B[!T]>) {
  trait.assoc_type @Out
  trait.method @b(!T) -> i64
}

trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {
  trait.method @a(!T) -> i64
  trait.method @twice(%x: !T) -> i64 {
    %b = trait.project %self[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
    %u = trait.method.call %self @A[!T]::@a(%x) : (!T) -> i64
    %v = trait.method.call %b @B[!T]::@b(%x) : (!T) -> i64
    %r = arith.addi %u, %v : i64
    trait.return %r : i64
  }
}

trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %out: !trait.claim<!trait.proj<@B[!T], "Out"> = i64>, %b_1: !trait.claim<@B[!T]>) {
  trait.method @a(%x: !T) -> i64 {
    %r = trait.method.call %b_1 @B[!T]::@b(%x) : (!T) -> i64
    trait.return %r : i64
  }
  trait.return %b : !trait.claim<@B[!T]>
}

// CHECK: trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) -> !trait.claim<@B[!trait.poly<0>]>
// CHECK: trait.project %self[0] : <@A[!trait.poly<0>]> -> <@B[!trait.poly<0>]>
// CHECK: trait.impl private @A_gen(%self: !trait.claim<@A[!trait.poly<0>]>, %b: !trait.claim<@B[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@B[!trait.poly<0>], "Out"> = i64>, %b_0: !trait.claim<@B[!trait.poly<0>]>)
// CHECK: trait.method.call %b_0 @B[!trait.poly<0>]::@b
// CHECK: trait.return %b : !trait.claim<@B[!trait.poly<0>]>
