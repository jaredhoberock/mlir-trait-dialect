// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// When a call declares its callee's type arguments, the verifier checks the
// actuals are an instance of that substitution, and refuses a type parameter
// that is not one of the callee's own type variables.
//
// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func private @foo(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
func.func @a(%x: i32) -> i32 {
  // The declared binding poly<0> := i64 makes the callee (i64) -> i64, which the
  // (i32) -> i32 actuals are not an instance of.
  // expected-error@+1 {{type mismatch: expected '(i64) -> i64' but found '(i32) -> i32'}}
  %r = trait.func.call @foo(%x) {type_params = [!trait.poly<0>], type_args = [i64]} : (i32) -> i32
  return %r : i32
}

// -----

func.func private @bar(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
func.func @b(%x: i64) -> i64 {
  // poly<9> is not a type variable of @bar.
  // expected-error@+1 {{is not a type variable of the callee}}
  %r = trait.func.call @bar(%x) {type_params = [!trait.poly<9>], type_args = [i64]} : (i64) -> i64
  return %r : i64
}
