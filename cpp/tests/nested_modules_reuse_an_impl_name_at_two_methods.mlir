// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// One impl name @T_i32 in two modules, whose methods answer two different
// values. A symbol name is read in the module the demand stands in and in no
// module around it, so each module specializes its own method and its own call
// reaches it. Which module is lowered first is the module order, and the answer
// is the same either way: the nested module stands first here and last below.

// The checks follow the order the specializations are printed in, which is the
// order the modules holding them stand in.
// CHECK: func.func private @T_i32_m
// CHECK: arith.constant 1
// CHECK: module @inner
// CHECK: func.func private @T_i32_m
// CHECK: arith.constant 2
// CHECK: func.func @main
// CHECK: call @T_i32_m
// CHECK: func.func @main
// CHECK: call @T_i32_m

trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
trait.impl private @T_i32 for @T[i32] {
  func.func @m() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
module @inner {
  trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
  trait.impl private @T_i32 for @T[i32] {
    func.func @m() -> i64 {
      %c = arith.constant 2 : i64
      return %c : i64
    }
  }
  func.func @main() -> i64 {
    %c = trait.allege @T[i32]
    %r = trait.method.call %c @T[i32]::@m() : () -> i64
    return %r : i64
  }
}
func.func @main() -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.method.call %c @T[i32]::@m() : () -> i64
  return %r : i64
}

// -----

// CHECK: func.func private @T_i32_m
// CHECK: arith.constant 1
// CHECK: func.func @main
// CHECK: call @T_i32_m
// CHECK: module @inner
// CHECK: func.func private @T_i32_m
// CHECK: arith.constant 2
// CHECK: func.func @main
// CHECK: call @T_i32_m

trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
trait.impl private @T_i32 for @T[i32] {
  func.func @m() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
func.func @main() -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.method.call %c @T[i32]::@m() : () -> i64
  return %r : i64
}
module @inner {
  trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
  trait.impl private @T_i32 for @T[i32] {
    func.func @m() -> i64 {
      %c = arith.constant 2 : i64
      return %c : i64
    }
  }
  func.func @main() -> i64 {
    %c = trait.allege @T[i32]
    %r = trait.method.call %c @T[i32]::@m() : () -> i64
    return %r : i64
  }
}
