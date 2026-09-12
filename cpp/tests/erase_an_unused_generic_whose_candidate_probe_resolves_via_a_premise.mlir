// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Resolving a projection in a generic signature reaches two conditional
// candidates for @Other[i64]. Only the @Mark[i32] premise is satisfiable;
// resolution succeeds and the unused template is erased.

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

// CHECK: module {
// CHECK-NEXT: }
