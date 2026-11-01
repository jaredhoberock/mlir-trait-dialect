// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The selected Choose member takes a Convert claim as a parameter. Propagating
// its proof must keep the function signature and body consistent and let the
// nested convert call lower.

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Convert(!trait.claim<@Convert[!T, !U]>) members [@Convert.convert]
func.func private @Convert.convert(!trait.claim<@Convert[!T, !U]>, !U) -> !T

trait.impl private @Convert_i32(%self: !trait.claim<@Convert[i32, i32]>) members [@Convert.convert -> @Convert_i32.convert] {
}
func.func private @Convert_i32.convert(%self: !trait.claim<@Convert[i32, i32]>, %x: i32) -> i32 {
  return %x : i32
}

trait.trait private @Choose(!trait.claim<@Choose[!T]>) members [@Choose.choose]
func.func private @Choose.choose(!trait.claim<@Choose[!T]>, !T, !trait.claim<@Convert[!T, !T]>) -> !T

trait.impl private @Choose_i32(%self: !trait.claim<@Choose[i32]>) members [@Choose.choose -> @Choose_i32.choose] {
}
func.func private @Choose_i32.choose(%self: !trait.claim<@Choose[i32]>, %a: i32, %same: !trait.claim<@Convert[i32, i32]>) -> i32 {
  %converted = trait.func.call @Convert.convert(%same, %a) : (!trait.claim<@Convert[i32, i32]>, i32) -> i32
  return %converted : i32
}

func.func @test(%x: i32) -> i32 {
  %chooser = trait.allege @Choose[i32]
  %same = trait.allege @Convert[i32, i32]
  %res = trait.func.call @Choose.choose(%chooser, %x, %same) : (!trait.claim<@Choose[i32]>, i32, !trait.claim<@Convert[i32, i32]>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func private @Convert_i32.convert_{{h[0-9a-f]+}}
// CHECK: return %arg0 : i32
// CHECK-LABEL: func.func private @Choose_i32.choose_{{h[0-9a-f]+}}
// CHECK: call @Convert_i32.convert_{{h[0-9a-f]+}}
// CHECK-LABEL: func.func @test
// CHECK: call @Choose_i32.choose_{{h[0-9a-f]+}}
