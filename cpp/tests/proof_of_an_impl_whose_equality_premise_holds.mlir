// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The twin of the refused proof: @Tensor_i8 binds Shape to i64, so
// @Vector_blanket's premise reads i64 = i64 at @Vector[i8] and the proof
// stands. The equality it carries is what the coerce spends.

trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
  func.func private @shape(!trait.poly<0>) -> !trait.proj<@Tensor[!trait.poly<0>], "Shape">
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = i64
  func.func @shape(%x: i8) -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}

trait.proof private @p proves @Vector_blanket for @Vector[i8] given [@Tensor_i8]

// CHECK: func.func @main
// CHECK: call @[[SHAPE:Tensor_i8_shape]]
// CHECK-NOT: trait.
func.func @main(%x: i8) -> i64 {
  %w = trait.witness @p for @Vector[i8]
  %t = trait.witness @Tensor_i8 for @Tensor[i8]
  %s = trait.method.call %t @Tensor[i8]::@shape(%x) : (i8) -> !trait.proj<@Tensor[i8], "Shape"> by @Tensor_i8
  %eq = trait.project %w[1] : !trait.claim<@Vector[i8] by @p> -> !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>
  %n = trait.coerce %s : !trait.proj<@Tensor[i8], "Shape"> to i64 via (%eq) : (!trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>)
  return %n : i64
}
