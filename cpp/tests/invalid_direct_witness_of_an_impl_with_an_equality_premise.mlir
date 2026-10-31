// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @I takes the premise that i32 is i64, which it is not. A witness names an
// impl directly only when the impl is unconditional -- it binds no parameter
// and has no where clause, so a citation supplies it nothing -- and @I's
// equality entry is a premise like any other, so naming it directly is
// refused.

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!trait.poly<0>]>) -> i64
trait.impl private @I(%self: !trait.claim<@T[i32]>, %eq: !trait.claim<i32 = i64>) members [@T.m -> @I.m] {
}
func.func private @I.m(%self: !trait.claim<@T[i32]>, %eq: !trait.claim<i32 = i64>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
func.func @main() -> i64 {
  // expected-error @below {{impl '@I' binds type parameters or has a where clause, so it must be cited through a trait.proof}}
  %w = trait.witness @I for @T[i32]
  %r = trait.func.call @T.m(%w) : (!trait.claim<@T[i32] by @I>) -> i64
  return %r : i64
}
