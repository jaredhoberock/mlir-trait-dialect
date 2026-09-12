// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Two satisfiable impls for @Gen[i64] provide different associated types.
// Neither may be chosen, so the surviving @Gen[i64]::A projection is diagnosed.

!T = !trait.poly<0>

trait.trait private @Gen[!T] {
  trait.assoc_type @A
}

trait.impl private @Gen_wide for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.impl private @Gen_narrow for @Gen[i64] {
  trait.assoc_type @A = i16
}

func.func private @wrap(%x: !T) -> !trait.proj<@Gen[!T], "A"> {
  %r = ub.poison : !trait.proj<@Gen[!T], "A">
  return %r : !trait.proj<@Gen[!T], "A">
}

func.func @main() -> !trait.proj<@Gen[i64], "A"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) {type_params = [!trait.poly<0>], type_args = [i64]} : (i64) -> !trait.proj<@Gen[i64], "A">
  return %r : !trait.proj<@Gen[i64], "A">
}
