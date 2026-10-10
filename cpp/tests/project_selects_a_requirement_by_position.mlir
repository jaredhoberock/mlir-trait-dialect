// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A hop names its requirement by position in the trait's `where` clause, and
// that clause mixes arms: the application at 0, the equality at 1 and the
// application at 2 are all reachable, and the claim each hop produces is the
// predicate at that position instantiated at the source claim's arguments -- an
// application claim for an application entry, an equality claim for an equality
// entry.

// RUN: mlir-opt %s | FileCheck %s

!S = !trait.poly<0>

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}

trait.trait private @Has(%self: !trait.claim<@Has[!S]>) -> (!trait.claim<@A[!S]>, !trait.claim<!trait.proj<@Has[!S], "Out"> = i64>, !trait.claim<@B[!S]>) {
  trait.assoc_type @Out
}

// CHECK-LABEL: func.func @f
// CHECK: trait.project %arg0[0] : <@Has[i32]> -> <@A[i32]>
// CHECK: trait.project %arg0[1] : <@Has[i32]> -> <!trait.proj<@Has[i32], "Out"> = i64>
// CHECK: trait.project %arg0[2] : <@Has[i32]> -> <@B[i32]>
func.func @f(%p: !trait.claim<@Has[i32]>) -> !trait.claim<@B[i32]> {
  %a = trait.project %p[0] : !trait.claim<@Has[i32]> -> !trait.claim<@A[i32]>
  %e = trait.project %p[1]
    : !trait.claim<@Has[i32]> -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
  %b = trait.project %p[2] : !trait.claim<@Has[i32]> -> !trait.claim<@B[i32]>
  return %b : !trait.claim<@B[i32]>
}
