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

trait.trait @ConstMapper[!S] {
  trait.assoc_type @Result<[!T]>
  func.func private @map(!S, !T) -> !trait.proj<@ConstMapper[!S], "Result", [!T]>
}

trait.impl for @ConstMapper[i32] {
  trait.assoc_type @Result<[!T]> = i64
  func.func @map(%self: i32, %x: !T) -> i64 {
    %c = arith.extsi %self : i32 to i64
    return %c : i64
  }
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
  %r = trait.method.call %a @ConstMapper[i32]::@map(%self, %x)
    : (i32, i1) -> !trait.proj<@ConstMapper[i32], "Result", [i1]>
  return %r : !trait.proj<@ConstMapper[i32], "Result", [i1]>
}
