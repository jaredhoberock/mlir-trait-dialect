// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The scope holds Tensor[i8]::Shape = i64 as an equality parameter, which is
// exactly @Vector_blanket's premise at i8. No impl of @Tensor stands in this
// module, so the hypothesis is the whole of what settles the premise.

trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}

// CHECK-LABEL: func.func private @f
// CHECK: trait.derive @Vector[i8] from @Vector_blanket
func.func private @f(%t: !trait.claim<@Tensor[i8]>,
                     %eq: !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>) {
  %v = trait.derive @Vector[i8] from @Vector_blanket given(%t) : (!trait.claim<@Tensor[i8]>)
  return
}
