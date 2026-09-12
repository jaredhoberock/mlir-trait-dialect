// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A template's interior is what instantiation skips and collection takes, so an
// uncalled template is read by nothing downstream: a type parameter its body
// mentions that its signature does not bind stands there unseen. The stage
// judges every function's body before it reads or rewrites one, so the
// declaration is named whether or not anything calls it.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// expected-error@+1 {{type parameter '!trait.poly<9>' is outside the signature scope of @unused}}
func.func private @unused(%x: !trait.poly<0>) -> !trait.poly<0> {
  // expected-note@+1 {{mentioned here}}
  %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<9>
  %b = builtin.unrealized_conversion_cast %a : !trait.poly<9> to !trait.poly<0>
  return %b : !trait.poly<0>
}
