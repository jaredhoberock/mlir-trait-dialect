// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation for @Greet[i32] supplies the evidence for the member call.
// The resulting module must contain the concrete greet function and a call to it.

!T = !trait.poly<0>

trait.trait private @Greet(!trait.claim<@Greet[!T]>) members [@Greet.greet]
func.func private @Greet.greet(!trait.claim<@Greet[!T]>, !T) -> i32

trait.impl private @Greet_i32(%self: !trait.claim<@Greet[i32]>) members [@Greet.greet -> @Greet_i32.greet] {
}
func.func private @Greet_i32.greet(%self: !trait.claim<@Greet[i32]>, %x: i32) -> i32 {
  return %x : i32
}

func.func @main(%x: i32) -> i32 {
  %w = trait.allege @Greet[i32]
  %r = trait.func.call @Greet.greet(%w, %x) : (!trait.claim<@Greet[i32]>, i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func private @Greet_i32.greet_{{h[0-9a-f]+}}
// CHECK: return %arg0 : i32
// CHECK-LABEL: func.func @main
// CHECK: call @Greet_i32.greet_{{h[0-9a-f]+}}
// CHECK: return {{.*}} : i32
