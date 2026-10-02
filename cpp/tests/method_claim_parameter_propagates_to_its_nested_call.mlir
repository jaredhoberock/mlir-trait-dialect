// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The selected Choose method takes a Convert claim as a parameter. Propagating
// its proof must keep the function signature and body consistent and let the
// nested convert call lower.

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Convert(%self: !trait.claim<@Convert[!T, !U]>) {
  trait.method @convert(!U) -> !T
}

trait.impl private @Convert_i32(%self: !trait.claim<@Convert[i32, i32]>) {
  trait.method @convert(%x: i32) -> i32 {
    trait.return %x : i32
  }
}

trait.trait private @Choose(%self: !trait.claim<@Choose[!T]>) {
  trait.method @choose(!T, !trait.claim<@Convert[!T, !T]>) -> !T
}

trait.impl private @Choose_i32(%self: !trait.claim<@Choose[i32]>) {
  trait.method @choose(%a: i32, %same: !trait.claim<@Convert[i32, i32]>) -> i32 {
    %converted = trait.method.call %same @Convert[i32, i32]::@convert(%a)
      : (i32) -> i32
    trait.return %converted : i32
  }
}

func.func @test(%x: i32) -> i32 {
  %chooser = trait.allege @Choose[i32]
  %same = trait.allege @Convert[i32, i32]
  %res = trait.method.call %chooser @Choose[i32]::@choose(%x, %same)
    : (i32, !trait.claim<@Convert[i32, i32]>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func private @Convert_i32_{{h[0-9a-f]+}}_convert
// CHECK: return %arg0 : i32
// CHECK-LABEL: func.func private @Choose_i32_{{h[0-9a-f]+}}_choose
// CHECK: call @Convert_i32_{{h[0-9a-f]+}}_convert
// CHECK-LABEL: func.func @test
// CHECK: call @Choose_i32_{{h[0-9a-f]+}}_choose
