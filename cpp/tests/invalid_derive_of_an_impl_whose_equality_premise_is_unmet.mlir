// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @Vector_blanket applies where Tensor[T]::Shape is i64. The derive holds the
// application hypothesis @Tensor[T] and nothing about T's shape, so the
// premise is one this scope does not meet. Reading the operands alone accepts
// it, and the clone at i8 is where it used to be caught.

trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = tuple<i64, i64>
}

func.func private @f(%t: !trait.claim<@Tensor[!trait.poly<0>]>) {
  // expected-error @below {{impl '@Vector_blanket' applies where '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' = 'i64', and nothing here makes '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' and 'i64' one type at '!trait.claim<@Vector[!trait.poly<0>]>'}}
  %v = trait.derive @Vector[!trait.poly<0>] from @Vector_blanket given(%t) : (!trait.claim<@Tensor[!trait.poly<0>]>)
  return
}
