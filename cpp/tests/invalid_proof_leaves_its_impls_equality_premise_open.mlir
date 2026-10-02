// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @OnlyI64 applies where its parameter is i64. @p stands over every instance of
// @Vector, so the premise reads over the proof's own variable and nothing in a
// closed proof body decides it: the evidence the proof states for it is
// refused at its own claim.

trait.trait private @Vector(%self: !trait.claim<@Vector[!trait.poly<0>]>) {
  trait.method @v() -> i64
}
trait.impl private @OnlyI64(%self: !trait.claim<@Vector[!trait.poly<0>]>, %eq: !trait.claim<!trait.poly<0> = i64>) {
  trait.method @v() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.proof private @p {
  // expected-error @below {{a refl witness requires identical endpoints, found '!trait.poly<0>' and 'i64'}}
  %e = trait.witness refl : !trait.claim<!trait.poly<0> = i64>
  %d = trait.derive @Vector[!trait.poly<0>] from @OnlyI64 given(%e) : (!trait.claim<!trait.poly<0> = i64>)
  trait.return %d : !trait.claim<@Vector[!trait.poly<0>]>
}
