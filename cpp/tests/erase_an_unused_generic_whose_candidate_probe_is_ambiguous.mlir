// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A generic signature contains a projection whose candidate header reaches
// an ambiguous associated type. The unused generic function is erased without
// turning that speculative lookup into a compilation failure.

!T = !trait.poly<0>

trait.trait private @Other(!trait.claim<@Other[!T]>) assoc {X = 0}

trait.impl private @Other_wide(%self: !trait.claim<@Other[i64]>) bindings {X = i32} {
}

trait.impl private @Other_narrow(%self: !trait.claim<@Other[i64]>) bindings {X = i16} {
}

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
