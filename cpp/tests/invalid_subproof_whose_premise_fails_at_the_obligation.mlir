// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @q discharges its obligation @Vector[i8] with @p, a proof standing over every
// instance of @Vector. @OnlyI64 applies only where its parameter is i64, and the
// premise the proof left to its instances is read at the obligation the
// subproof discharges.

trait.trait private @Vector[!trait.poly<0>] { func.func private @v() -> i64 }
trait.trait private @Uses[!trait.poly<0>] where [@Vector[!trait.poly<0>]] {
  func.func private @u() -> i64
}
trait.impl private @OnlyI64 for @Vector[!trait.poly<0>]
    where [!trait.poly<0> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @Uses_i8 for @Uses[i8] {
  func.func @u() -> i64 {
    %c = arith.constant 3 : i64
    return %c : i64
  }
}
trait.proof private @p proves @OnlyI64 for @Vector[!trait.poly<0>] given []
// expected-error @below {{impl '@OnlyI64' applies where '!trait.poly<0>' = 'i64', and nothing here makes 'i8' and 'i64' one type at '!trait.claim<@Vector[i8]>'}}
trait.proof private @q proves @Uses_i8 for @Uses[i8] given [@p]
