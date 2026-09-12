// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// The same body, cloned for a concrete call: the clone's signature binds no type
// parameter at all, and the pair of casts carrying the stray one cancels under
// the driver's folding, so nothing afterwards can tell the clone from a
// well-scoped one. The judgment runs at the entry, before any round folds or
// clones, and names the declaration the clone would come from.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// expected-error@+1 {{type parameter '!trait.poly<9>' is outside the signature scope of @called}}
func.func private @called(%x: !trait.poly<0>) -> !trait.poly<0> {
  // expected-note@+1 {{mentioned here}}
  %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<9>
  %b = builtin.unrealized_conversion_cast %a : !trait.poly<9> to !trait.poly<0>
  return %b : !trait.poly<0>
}

func.func @host(%v: i64) -> i64 {
  %r = trait.func.call @called(%v) : (i64) -> i64
  return %r : i64
}
