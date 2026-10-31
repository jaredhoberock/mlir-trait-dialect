// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests GAT with a non-identity binding: the GAT parameter does not appear
// in the bound type.
//
// trait ConstMapper {
//   type Result<T>;
//   fn map<T>(self: Self, x: T) -> Self::Result<T>;
// }
// impl ConstMapper for i32 {
//   type Result<T> = i64;   // ignores T entirely
//   fn map<T>(self: i32, x: T) -> i64 { self as i64 }
// }

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @ConstMapper(!trait.claim<@ConstMapper[!S]>) assoc {Result = 1} members [@ConstMapper.map]
func.func private @ConstMapper.map(!trait.claim<@ConstMapper[!S]>, !S, !T) -> !trait.proj<@ConstMapper[!S], "Result", [!T]>

trait.impl private @ConstMapper_impl(%self_claim: !trait.claim<@ConstMapper[i32]>) bindings {Result = i64} members [@ConstMapper.map -> @ConstMapper_impl.map] {
}
func.func private @ConstMapper_impl.map(%self_claim: !trait.claim<@ConstMapper[i32]>, %self: i32, %x: !trait.poly<0>) -> i64 {
  %c = arith.extsi %self : i32 to i64
  return %c : i64
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: !trait.proj
// CHECK: call @
// CHECK-SAME: (i32, i1) -> i64
// CHECK: return %{{.*}} : i64
func.func @caller() -> !trait.proj<@ConstMapper[i32], "Result", [i1]> {
  %a = trait.allege @ConstMapper[i32]
  %self = arith.constant 7 : i32
  %x = arith.constant 1 : i1
  %r = trait.func.call @ConstMapper.map(%a, %self, %x) : (!trait.claim<@ConstMapper[i32]>, i32, i1) -> !trait.proj<@ConstMapper[i32], "Result", [i1]>
  return %r : !trait.proj<@ConstMapper[i32], "Result", [i1]>
}
