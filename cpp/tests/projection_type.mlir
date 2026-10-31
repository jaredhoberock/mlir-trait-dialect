// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: trait.trait private @Get(!trait.claim<@Get[!trait.poly<0>]>){{$}}
// CHECK-NEXT: assoc {Output = 0}
// CHECK-NEXT: members [
// CHECK-NEXT: @Get.get{{$}}
// CHECK-NEXT: ]
// CHECK: func.func private @Get.get(!trait.claim<@Get[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Get[!trait.poly<0>], "Output">

!T = !trait.poly<0>
trait.trait private @Get(!trait.claim<@Get[!T]>) assoc {Output = 0} members [@Get.get]
func.func private @Get.get(!trait.claim<@Get[!T]>, !T) -> !trait.proj<@Get[!T], "Output">

// CHECK: func.func @use_proj
// CHECK-SAME: !trait.proj<@Get[i32], "Output">
func.func @use_proj(%x: !trait.proj<@Get[i32], "Output">) -> !trait.proj<@Get[i32], "Output"> {
  return %x : !trait.proj<@Get[i32], "Output">
}
