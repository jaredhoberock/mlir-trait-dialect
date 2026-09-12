// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The twin of the refused derive: the scope holds Tensor[T]::Shape = i64 as an
// equality parameter, which is exactly @Vector_blanket's premise at T, so the
// derive stands.

trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}

// CHECK-LABEL: func.func private @f
// CHECK: trait.derive @Vector[!trait.poly<0>] from @Vector_blanket
func.func private @f(%t: !trait.claim<@Tensor[!trait.poly<0>]>,
                     %eq: !trait.claim<!trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64>) {
  %v = trait.derive @Vector[!trait.poly<0>] from @Vector_blanket given(%t) : (!trait.claim<@Tensor[!trait.poly<0>]>)
  return
}
