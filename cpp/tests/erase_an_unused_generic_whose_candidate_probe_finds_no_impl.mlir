// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Matching @Gen_via probes a projection for which @Other has no impl. The
// unused polymorphic declaration must still compile and disappear during erasure.

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

func.func private @f(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i32>,
                     %x: !T) -> !T {
  return %x : !T
}

// CHECK: module {
// CHECK-NEXT: }
