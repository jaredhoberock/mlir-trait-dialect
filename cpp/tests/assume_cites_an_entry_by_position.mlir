// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// An impl's member function leads with the impl's block arguments -- the self
// claim, then one claim per where entry -- and cites them by position, each
// typed as the claim the entry states; a trait's default reads a requirement
// off its self claim by index. The impl's first and third entries spell one
// claim and stay two arguments: its member function reads the third and its
// return the first.

!T = !trait.poly<0>

trait.trait private @B(!trait.claim<@B[!T]>) assoc {Out = 0} members [@B.b]
func.func private @B.b(!trait.claim<@B[!T]>, !T) -> i64

trait.trait private @A(!trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> members [@A.a, default @A.twice]
func.func private @A.a(!trait.claim<@A[!T]>, !T) -> i64
func.func private @A.twice(%self: !trait.claim<@A[!T]>, %x: !T) -> i64 {
  %b = trait.project %self[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
  %u = trait.func.call @A.a(%self, %x) : (!trait.claim<@A[!T]>, !T) -> i64
  %v = trait.func.call @B.b(%b, %x) : (!trait.claim<@B[!T]>, !T) -> i64
  %r = arith.addi %u, %v : i64
  return %r : i64
}

trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %out: !trait.claim<!trait.proj<@B[!T], "Out"> = i64>, %b_1: !trait.claim<@B[!T]>) members [@A.a -> @A_gen.a] {
  trait.return %b : !trait.claim<@B[!T]>
}
func.func private @A_gen.a(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %out: !trait.claim<!trait.proj<@B[!T], "Out"> = i64>, %b_1: !trait.claim<@B[!T]>, %x: !T) -> i64 {
  %r = trait.func.call @B.b(%b_1, %x) : (!trait.claim<@B[!T]>, !T) -> i64
  return %r : i64
}

// CHECK: trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) -> !trait.claim<@B[!trait.poly<0>]>{{$}}
// CHECK-NEXT: members [
// CHECK-NEXT: @A.a,
// CHECK-NEXT: default @A.twice{{$}}
// CHECK-NEXT: ]
// CHECK: func.func private @A.twice(%[[SELF:[^:]*]]: !trait.claim<@A[!trait.poly<0>]>
// CHECK: trait.project %[[SELF]][0] : <@A[!trait.poly<0>]> -> <@B[!trait.poly<0>]>
// CHECK: trait.impl private @A_gen(%self: !trait.claim<@A[!trait.poly<0>]>, %b: !trait.claim<@B[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@B[!trait.poly<0>], "Out"> = i64>, %b_0: !trait.claim<@B[!trait.poly<0>]>)
// CHECK: trait.return %b : !trait.claim<@B[!trait.poly<0>]>
// CHECK: func.func private @A_gen.a(%{{[^:]*}}: !trait.claim<@A[!trait.poly<0>]>, %{{[^:]*}}: !trait.claim<@B[!trait.poly<0>]>, %{{[^:]*}}: !trait.claim<!trait.proj<@B[!trait.poly<0>], "Out"> = i64>, %[[THIRD:[^:]*]]: !trait.claim<@B[!trait.poly<0>]>, %[[X:[^:]*]]: !trait.poly<0>)
// CHECK: trait.func.call @B.b(%[[THIRD]], %[[X]])
