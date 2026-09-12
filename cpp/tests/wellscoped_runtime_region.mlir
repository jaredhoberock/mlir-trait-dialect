// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A region an op runs at run time declares no type parameters of its own: it is
// interior to the function around it and stands in that function's scope. A
// conditional whose results and yields carry the function's own parameter is
// accepted, and one that reaches for a parameter the function does not bind is
// named at the function, so the judgment reads inside the region rather than
// stopping at it.

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

func.func private @in_scope(%c: i1, %x: !trait.poly<0>) -> !trait.poly<0> {
  %r = scf.if %c -> (!trait.poly<0>) {
    scf.yield %x : !trait.poly<0>
  } else {
    %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<0>
    scf.yield %a : !trait.poly<0>
  }
  return %r : !trait.poly<0>
}

// -----

// expected-error@+1 {{type parameter '!trait.poly<7>' is outside the signature scope of @out_of_scope}}
func.func private @out_of_scope(%c: i1, %x: !trait.poly<0>) -> !trait.poly<0> {
  %r = scf.if %c -> (!trait.poly<0>) {
    scf.yield %x : !trait.poly<0>
  } else {
    // expected-note@+1 {{mentioned here}}
    %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<7>
    %b = builtin.unrealized_conversion_cast %a : !trait.poly<7> to !trait.poly<0>
    scf.yield %b : !trait.poly<0>
  }
  return %r : !trait.poly<0>
}
