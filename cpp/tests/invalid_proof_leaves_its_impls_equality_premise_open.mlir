// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @OnlyI64 applies where its parameter is i64. @p stands over every instance of
// @Vector, so the premise reads over the proof's own variable and this claim
// does not decide it. A citation of @p reads nothing inside it, so no instance
// reads that premise either: the proof is refused at its own claim.

trait.trait private @Vector[!trait.poly<0>] {
  func.func private @v() -> i64
}
trait.impl private @OnlyI64 for @Vector[!trait.poly<0>]
    where [!trait.poly<0> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
// expected-error @below {{a proof states its impl's premises at its own claim; one the claim leaves open is stated at the instance instead: '!trait.poly<0>' = 'i64' reads '!trait.poly<0>' = 'i64' at '!trait.claim<@Vector[!trait.poly<0>] by @p>'}}
trait.proof private @p proves @OnlyI64 for @Vector[!trait.poly<0>] given []
