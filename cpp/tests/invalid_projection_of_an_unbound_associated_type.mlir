// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Selecting @Gen_i64 cannot resolve the undeclared associated type B.
// The surviving @Gen[i64]::B projection must be diagnosed even though the
// trait application itself has a unique implementation.

!T = !trait.poly<0>

trait.trait private @Gen[!T] {
  trait.assoc_type @A
}

trait.impl private @Gen_i64 for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.trait private @Box[!T] {}

trait.impl private @Box_i32 for @Box[i32] {}

func.func private @probes(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "B">]>,
                          %x: !T) -> !T {
  return %x : !T
}

func.func private @wrap(%x: !T) -> !trait.proj<@Gen[!T], "B"> {
  %r = ub.poison : !trait.proj<@Gen[!T], "B">
  return %r : !trait.proj<@Gen[!T], "B">
}

func.func @main() -> !trait.proj<@Gen[i64], "B"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "B">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) {type_params = [!trait.poly<0>], type_args = [i64]} : (i64) -> !trait.proj<@Gen[i64], "B">
  return %r : !trait.proj<@Gen[i64], "B">
}
