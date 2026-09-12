// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @Vector_blanket applies where Tensor[T]::Shape is i64. @Tensor_i8 binds
// Shape to tuple<i64, i64>, so the impl does not apply at i8 and no proof
// stands over @Vector[i8]. The premise takes no subproof -- the given list is
// indexed by the impl's application-arm obligations -- so the proof must read
// it through the evidence it cites rather than leave it to a later use.

trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = tuple<i64, i64>
}

// expected-error @below {{impl '@Vector_blanket' applies where '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' = 'i64', which at '!trait.claim<@Vector[i8] by @p>' reads 'tuple<i64, i64>' = 'i64'}}
trait.proof private @p proves @Vector_blanket for @Vector[i8] given [@Tensor_i8]
