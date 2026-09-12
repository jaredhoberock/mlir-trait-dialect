// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A blanket Box impl can prove a claim whose argument contains an unresolved
// projection. That proof does not resolve @T[i64]::A, so the projection must
// still be diagnosed on the allegation that carries it.

!T = !trait.poly<0>

trait.trait private @T[!T] {
  trait.assoc_type @A
}

trait.trait private @Box[!T] {}

trait.impl private @Box_any for @Box[!T] {}

func.func private @callee(%c: !trait.claim<@Box[!trait.proj<@T[i64], "A">]>,
                  %x: !T) -> !T {
  return %x : !T
}

func.func @main() -> i64 {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %c = trait.allege @Box[!trait.proj<@T[i64], "A">]
  %x = arith.constant 0 : i64
  %r = trait.func.call @callee(%c, %x) {type_params = [!trait.poly<0>], type_args = [i64]}
    : (!trait.claim<@Box[!trait.proj<@T[i64], "A">]>, i64) -> i64
  return %r : i64
}
