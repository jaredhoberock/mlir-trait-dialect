// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Matching @Gen_via probes a projection for which @Other has no impl. The
// unused polymorphic declaration must still compile and disappear during erasure.

!T = !trait.poly<0>

trait.trait private @Other(!trait.claim<@Other[!T]>) assoc {X = 0}

trait.trait private @Gen(!trait.claim<@Gen[!T]>) assoc {A = 0}

trait.impl private @Gen_via(%self: !trait.claim<@Gen[!trait.proj<@Other[i64], "X">]>) bindings {A = i32} {
}

trait.trait private @Box(!trait.claim<@Box[!T]>)

trait.impl private @Box_i32(%self: !trait.claim<@Box[i32]>) {
}

func.func private @f(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>,
                     %x: !T) -> !T {
  return %x : !T
}

// CHECK: module {
// CHECK-NEXT: }
