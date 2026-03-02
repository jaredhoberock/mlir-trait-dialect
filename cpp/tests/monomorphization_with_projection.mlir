// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T = !trait.poly<0>

trait.trait @Get[!T] {
  trait.assoc_type @Output
  func.func private @get(!T) -> !trait.proj<@Get[!T], "Output">
}

trait.impl for @Get[i32] {
  trait.assoc_type @Output = i64
  func.func @get(%self: i32) -> i64 {
    %c = arith.extsi %self : i32 to i64
    return %c : i64
  }
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: !trait.proj
// CHECK: call @
// CHECK-SAME: (i32) -> i64
// CHECK: return %{{.*}} : i64
func.func @caller() -> !trait.proj<@Get[i32], "Output"> {
  %a = trait.allege @Get[i32]
  %x = arith.constant 7 : i32
  %r = trait.method.call %a @Get[i32]::@get(%x)
    : (i32) -> !trait.proj<@Get[i32], "Output">
  return %r : !trait.proj<@Get[i32], "Output">
}

// A free function with projections in argument and return types
// CHECK-LABEL: func.func @identity
// CHECK-SAME: (%{{.*}}: i64) -> i64
// CHECK-NOT: !trait.proj
// CHECK: return %{{.*}} : i64
func.func @identity(%x: !trait.proj<@Get[i32], "Output">) -> !trait.proj<@Get[i32], "Output"> {
  return %x : !trait.proj<@Get[i32], "Output">
}
