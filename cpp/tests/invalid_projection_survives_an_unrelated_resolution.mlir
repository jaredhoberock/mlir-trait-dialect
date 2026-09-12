// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An unsatisfied conditional candidate does not prevent the other candidate from
// resolving. The unrelated @Absent[i64]::B projection remains unresolved and
// must retain its diagnostic after specialization.

!T = !trait.poly<0>

trait.trait private @Mark[!T] {}

trait.impl private @Mark_i32 for @Mark[i32] {}

trait.trait private @Other[!T] {
  trait.assoc_type @X
}

trait.impl private @Other_wide for @Other[i64] where [@Mark[i32]] {
  trait.assoc_type @X = i32
}

trait.impl private @Other_narrow for @Other[i64] where [@Mark[i16]] {
  trait.assoc_type @X = i16
}

trait.trait private @Gen[!T] {
  trait.assoc_type @A
}

trait.impl private @Gen_via for @Gen[!trait.proj<@Other[i64], "X">] {
  trait.assoc_type @A = i32
}

trait.trait private @Box[!T] {}

trait.impl private @Box_i32 for @Box[i32] {}

func.func private @f(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>,
                     %x: !T) -> !T {
  return %x : !T
}

trait.trait private @Absent[!T] {
  trait.assoc_type @B
}

func.func private @wrap(%x: !T) -> !trait.proj<@Absent[!T], "B"> {
  %r = ub.poison : !trait.proj<@Absent[!T], "B">
  return %r : !trait.proj<@Absent[!T], "B">
}

func.func @main() -> !trait.proj<@Absent[i64], "B"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Absent[i64], "B">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) {type_params = [!trait.poly<0>], type_args = [i64]} : (i64) -> !trait.proj<@Absent[i64], "B">
  return %r : !trait.proj<@Absent[i64], "B">
}
