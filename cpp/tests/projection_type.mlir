// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: trait @Get
// CHECK: trait.assoc_type @Output
// CHECK: func.func private @get(!trait.poly<0>) -> !trait.proj<@Get[!trait.poly<0>], "Output">

!T = !trait.poly<0>
trait.trait @Get[!T] {
  trait.assoc_type @Output
  func.func private @get(!T) -> !trait.proj<@Get[!T], "Output">
}

// CHECK: func.func @use_proj
// CHECK-SAME: !trait.proj<@Get[i32], "Output">
func.func @use_proj(%x: !trait.proj<@Get[i32], "Output">) -> !trait.proj<@Get[i32], "Output"> {
  return %x : !trait.proj<@Get[i32], "Output">
}
