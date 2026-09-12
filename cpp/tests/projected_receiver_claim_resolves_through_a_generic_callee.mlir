// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A method receiver claim carries @Outer[i64]::Item through a generic call.
// Resolving the associated type must let both the generic call and its method
// call lower to the concrete i64 implementation.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Trait[!S, !T] {
  func.func private @method(!S, !T) -> i64
}

trait.trait private @Outer[!S] {
  trait.assoc_type @Item
}

func.func private @callee(%t: !T,
    %outer: !trait.claim<@Outer[i64]>,
    %claim: !trait.claim<@Trait[!T, !trait.proj<@Outer[i64], "Item">]>) -> i64 {
  %x = arith.constant 1 : i64
  %px = trait.coerce %x : i64 to !trait.proj<@Outer[i64], "Item"> unproven
  %result = trait.method.call %claim @Trait[!T, !trait.proj<@Outer[i64], "Item">]::@method(%t, %px)
    : (!T, !trait.proj<@Outer[i64], "Item">) -> i64
  return %result : i64
}

trait.impl private @Outer_i64 for @Outer[i64] {
  trait.assoc_type @Item = i64
}

trait.impl private @Trait_i64 for @Trait[i64, i64] {
  func.func @method(%self: i64, %x: i64) -> i64 {
    return %x : i64
  }
}

func.func @main() -> i64 {
  %outer = trait.witness @Outer_i64 for @Outer[i64]
  %trait = trait.witness @Trait_i64 for @Trait[i64, i64]
  %eq = trait.witness proj_resolve !trait.proj<@Outer[i64], "Item"> resolves i64 by @Outer_i64
    : !trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>
  %projected = trait.coerce %trait
    : !trait.claim<@Trait[i64, i64] by @Trait_i64>
    to !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>
    via (%eq) : (!trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>)
  %x = arith.constant 0 : i64
  %result = trait.func.call @callee(%x, %outer, %projected) {type_params = [!trait.poly<1>], type_args = [i64]}
    : (i64, !trait.claim<@Outer[i64] by @Outer_i64>,
       !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>) -> i64
  return %result : i64
}

// CHECK-LABEL: func.func private @callee_
// CHECK: call @Trait_i64_method
// CHECK-LABEL: func.func private @Trait_i64_method
// CHECK: return %arg1 : i64
// CHECK-LABEL: func.func @main() -> i64
// CHECK: call @callee_
// CHECK: return {{.*}} : i64
