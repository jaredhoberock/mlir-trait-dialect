// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @I applies where i32 is i64, which it is not. It binds no type parameter and
// assumes no application, so a citation may name it directly; the premise it
// does state is read at the obligation the citation discharges, which is the
// only place it can be decided.

trait.trait private @T[!trait.poly<0>] {
  func.func private @m() -> i64
}
trait.impl private @I for @T[i32] where [i32 = i64] {
  func.func @m() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
trait.trait private @Uses[!trait.poly<0>] where [@T[!trait.poly<0>]] {
  func.func private @u() -> i64
}
trait.impl private @Uses_i32 for @Uses[i32] {
  func.func @u() -> i64 {
    %c = arith.constant 3 : i64
    return %c : i64
  }
}
// expected-error @below {{impl '@I' applies where 'i32' = 'i64', and nothing here makes 'i32' and 'i64' one type at '!trait.claim<@T[i32]>'}}
trait.proof private @q proves @Uses_i32 for @Uses[i32] given [@I]
