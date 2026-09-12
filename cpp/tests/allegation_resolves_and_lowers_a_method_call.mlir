// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation for @Greet[i32] supplies the evidence for the method call.
// The resulting module must contain the concrete greet function and a call to it.

!T = !trait.poly<0>

trait.trait private @Greet[!T] {
  func.func private @greet(!T) -> i32
}

trait.impl private @Greet_i32 for @Greet[i32] {
  func.func @greet(%x: i32) -> i32 {
    return %x : i32
  }
}

func.func @main(%x: i32) -> i32 {
  %w = trait.allege @Greet[i32]
  %r = trait.method.call %w @Greet[i32]::@greet(%x) : (i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func private @Greet_i32_greet
// CHECK: return %arg0 : i32
// CHECK-LABEL: func.func @main
// CHECK: call @Greet_i32_greet
// CHECK: return {{.*}} : i32
