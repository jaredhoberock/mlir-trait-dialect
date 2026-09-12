// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// @Vector_blanket applies where Tensor[T]::Shape is i64. The premise takes no
// operand -- the operand list is indexed by the impl's application-arm
// assumptions -- so the derive reads it through the evidence in hand, rather
// than leaving a later use to report a claim nothing can settle.

// Read through the impls the module holds: at i8 the shape is tuple<i64, i64>.
trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = tuple<i64, i64>
}

func.func private @f(%t: !trait.claim<@Tensor[i8]>) {
  // expected-error @below {{impl '@Vector_blanket' applies where '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' = 'i64', and nothing here makes 'tuple<i64, i64>' and 'i64' one type at '!trait.claim<@Vector[i8]>'}}
  %v = trait.derive @Vector[i8] from @Vector_blanket given(%t) : (!trait.claim<@Tensor[i8]>)
  return
}

// -----

// Read through the hypotheses the scope holds: this one says the shape is f32,
// which is a premise this derive meets nowhere.
trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}

func.func private @g(%t: !trait.claim<@Tensor[i8]>,
                     %eq: !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = f32>) {
  // expected-error @below {{impl '@Vector_blanket' applies where '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' = 'i64', and nothing here makes 'f32' and 'i64' one type at '!trait.claim<@Vector[i8]>'}}
  %v = trait.derive @Vector[i8] from @Vector_blanket given(%t) : (!trait.claim<@Tensor[i8]>)
  return
}
