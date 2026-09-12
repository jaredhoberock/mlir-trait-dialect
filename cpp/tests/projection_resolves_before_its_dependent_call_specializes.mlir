// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The Outer associated type resolves before specializing the Sink call.
// The resulting concrete callee and call site must both use i64.

!T = !trait.poly<0>

trait.trait private @Outer[!T] {
  trait.assoc_type @Item
}

trait.impl private @Outer_i64 for @Outer[i64] {
  trait.assoc_type @Item = i64
}

trait.trait private @Sink[!T] {}

trait.impl private @Sink_any for @Sink[!T] {}

func.func private @callee(%c: !trait.claim<@Sink[!trait.proj<@Outer[i64], "Item">]>,
                  %x: !T) -> !T {
  return %x : !T
}

func.func @main() -> i64 {
  %c = trait.allege @Sink[!trait.proj<@Outer[i64], "Item">]
  %x = arith.constant 0 : i64
  %r = trait.func.call @callee(%c, %x) {type_params = [!trait.poly<0>], type_args = [i64]}
    : (!trait.claim<@Sink[!trait.proj<@Outer[i64], "Item">]>, i64) -> i64
  return %r : i64
}

// CHECK-LABEL: func.func private @callee_
// CHECK: return %arg0 : i64
// CHECK-LABEL: func.func @main() -> i64
// CHECK: call @callee_
// CHECK: return {{.*}} : i64
