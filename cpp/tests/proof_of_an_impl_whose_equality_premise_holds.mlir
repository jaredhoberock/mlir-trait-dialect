// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The twin of the refused proof: @Tensor_i8 binds Shape to i64, so the proof
// supplies @Vector_blanket's equality entry at @Vector[i8] with a proj_resolve
// witness citing @Tensor_i8, and the proof stands. The equality its derive
// carries is what the coerce spends.

trait.trait private @Tensor(%self: !trait.claim<@Tensor[!trait.poly<0>]>) {
  trait.assoc_type @Shape
  trait.method @shape(!trait.poly<0>) -> !trait.proj<@Tensor[!trait.poly<0>], "Shape">
}
trait.trait private @Vector(%self: !trait.claim<@Vector[!trait.poly<0>]>) {}
trait.impl private @Vector_blanket(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %shape: !trait.claim<!trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64>) {}
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) {
  trait.assoc_type @Shape = i64
  trait.method @shape(%x: i8) -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

trait.proof private @p {
  %p0 = trait.witness @Tensor_i8 for @Tensor[i8]
  %p1 = trait.witness proj_resolve !trait.proj<@Tensor[i8], "Shape"> resolves i64 by @Tensor_i8
    : !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>
  %d = trait.derive @Vector[i8] from @Vector_blanket given(%p0, %p1) : (!trait.claim<@Tensor[i8] by @Tensor_i8>, !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>)
  trait.return %d : !trait.claim<@Vector[i8]>
}

// CHECK: func.func @main
// CHECK: call @[[SHAPE:Tensor_i8_h[0-9a-f]+_shape]]
// CHECK-NOT: trait.
func.func @main(%x: i8) -> i64 {
  %w = trait.witness @p for @Vector[i8]
  %t = trait.witness @Tensor_i8 for @Tensor[i8]
  %s = trait.method.call %t @Tensor[i8]::@shape(%x) : (i8) -> !trait.proj<@Tensor[i8], "Shape"> by @Tensor_i8
  %eq = trait.project %w[1] : !trait.claim<@Vector[i8] by @p> -> !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>
  %n = trait.coerce %s : !trait.proj<@Tensor[i8], "Shape"> to i64 via (%eq) : (!trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>)
  return %n : i64
}
