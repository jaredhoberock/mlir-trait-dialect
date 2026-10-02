// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @Vector_blanket applies where Tensor[T]::Shape is i64. @Tensor_i8 binds
// Shape to tuple<i64, i64>, so the impl does not apply at i8 and no proof
// stands over @Vector[i8]: the evidence the proof holds for the equality
// premise resolves the projection to tuple<i64, i64>, which is not the premise
// the impl states at i8.

trait.trait private @Tensor(%self: !trait.claim<@Tensor[!trait.poly<0>]>) {
  trait.assoc_type @Shape
}
trait.trait private @Vector(%self: !trait.claim<@Vector[!trait.poly<0>]>) {}
trait.impl private @Vector_blanket(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %shape: !trait.claim<!trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64>) {}
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) {
  trait.assoc_type @Shape = tuple<i64, i64>
}

trait.proof private @p {
  %t = trait.witness @Tensor_i8 for @Tensor[i8]
  %s = trait.witness proj_resolve !trait.proj<@Tensor[i8], "Shape"> resolves tuple<i64, i64> by @Tensor_i8 : !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = tuple<i64, i64>>
  // expected-error @below {{premise 1 of impl '@Vector_blanket' is '!trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>', and the derive supplies '!trait.claim<!trait.proj<@Tensor[i8], "Shape"> = tuple<i64, i64>>'}}
  %d = trait.derive @Vector[i8] from @Vector_blanket given(%t, %s) : (!trait.claim<@Tensor[i8] by @Tensor_i8>, !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = tuple<i64, i64>>)
  trait.return %d : !trait.claim<@Vector[i8]>
}
