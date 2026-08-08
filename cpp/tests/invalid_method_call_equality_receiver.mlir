// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// trait.method.call names the trait to call through the receiver claim's trait
// application, so the receiver must be a trait-application claim. An equality
// claim names no trait, so it is refused at the operand as a verify diagnostic
// rather than asserting inside the arm-asserting accessor.

trait.trait @Trait[!trait.poly<0>] {
  func.func private @m(!trait.poly<0>) -> i1
}

func.func @f(%x: i64) -> i1 {
  %e = trait.witness refl : !trait.claim<i64 = i64>
  // expected-error @below {{operand #0 must be a trait-application '!trait.claim' type, but got '!trait.claim<i64 = i64>'}}
  %r = "trait.method.call"(%e, %x) <{method_ref = @Trait::@m}> : (!trait.claim<i64 = i64>, i64) -> i1
  return %r : i1
}
