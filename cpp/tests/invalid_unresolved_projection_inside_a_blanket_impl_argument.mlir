// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A blanket Box impl matches any argument, but a claim whose argument is a
// projection no impl resolves is about no type: selection refuses it, named by
// why @T[i64]::A names no type, rather than prove it over the projection as an
// opaque type, and the allegation that carries it stands unproven.

!T = !trait.poly<0>

trait.trait private @T(%self: !trait.claim<@T[!T]>) {
  trait.assoc_type @A
}

trait.trait private @Box(%self: !trait.claim<@Box[!T]>) {}

trait.impl private @Box_any(%self: !trait.claim<@Box[!T]>) {}

func.func private @callee(%c: !trait.claim<@Box[!trait.proj<@T[i64], "A">]>,
                  %x: !T) -> !T {
  return %x : !T
}

func.func @main() -> i64 {
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@T[i64], "A">'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@Box[!trait.proj<@T[i64], "A">]>' after instantiate-monomorphs}}
  %c = trait.allege @Box[!trait.proj<@T[i64], "A">]
  %x = arith.constant 0 : i64
  %r = trait.func.call @callee(%c, %x)
    : (!trait.claim<@Box[!trait.proj<@T[i64], "A">]>, i64) -> i64
  return %r : i64
}
