// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A function whose signature spells no type parameter is no template: the stage
// walks its body and the erase barrier judges what stands there. Neither sees
// this parameter, because the two casts carrying it cancel; the well-scopedness
// judgment reads the body as written and names it.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// expected-error@+1 {{type parameter '!trait.poly<4>' is outside the signature scope of @monomorphic}}
func.func @monomorphic(%x: i64) -> i64 {
  // expected-note@+1 {{mentioned here}}
  %a = builtin.unrealized_conversion_cast %x : i64 to !trait.poly<4>
  %b = builtin.unrealized_conversion_cast %a : !trait.poly<4> to i64
  return %b : i64
}
