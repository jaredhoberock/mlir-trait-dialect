// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A template cloned for a concrete call: the clone binds no type variable of its
// own, so every variable the body it copies spells must come from the call's
// substitution. One the substitution binds nothing for has no argument to
// receive and would stand in ground code as written, so the stamp-out is refused
// at the declaration and no clone is cut; the call is left standing for the exit
// check to name. Nothing later could tell: the pair of casts carrying the
// variable cancels under the driver's folding.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// expected-error@+1 {{type variable '!trait.poly<9>' in the body of '@called' is bound by no parameter of its declaration, so no instance can replace it}}
func.func private @called(%x: !trait.poly<0>) -> !trait.poly<0> {
  // expected-note@+1 {{mentioned here}}
  %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<9>
  %b = builtin.unrealized_conversion_cast %a : !trait.poly<9> to !trait.poly<0>
  return %b : !trait.poly<0>
}

func.func @host(%v: i64) -> i64 {
  // expected-error@+1 {{rewritable generic call survived instantiate-monomorphs}}
  %r = trait.func.call @called(%v) {type_params = [!trait.poly<0>], type_args = [i64]} : (i64) -> i64
  return %r : i64
}
