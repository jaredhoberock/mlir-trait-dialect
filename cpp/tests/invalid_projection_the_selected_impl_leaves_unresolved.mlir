// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Selecting @Gen_via does not supply an impl for @Other[i64]. The unresolved
// @Other[i64]::X projection must retain its diagnostic.

!T = !trait.poly<0>

trait.trait private @Other[!T] {
  trait.assoc_type @X
}

trait.trait private @Gen[!T] {
  trait.assoc_type @A
}

trait.impl private @Gen_via for @Gen[!trait.proj<@Other[i64], "X">] {
  trait.assoc_type @A = i32
}

trait.trait private @Box[!T] {}

trait.impl private @Box_i32 for @Box[i32] {}

func.func private @reads(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>,
                         %x: !T) -> !T {
  return %x : !T
}

func.func @asks() -> !trait.proj<@Other[i64], "X"> {
  // expected-error @below {{unresolved projection '!trait.proj<@Other[i64], "X">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Other[i64], "X">
  return %r : !trait.proj<@Other[i64], "X">
}
