// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A member receiver claim carries @Outer[i64]::Item through a generic call.
// Resolving the associated type must let both the generic call and its member
// call lower to the concrete i64 implementation.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Trait(!trait.claim<@Trait[!S, !T]>) members [@Trait.method]
func.func private @Trait.method(!trait.claim<@Trait[!S, !T]>, !S, !T) -> i64

trait.trait private @Outer(!trait.claim<@Outer[!S]>) assoc {Item = 0}

func.func private @callee(%t: !trait.poly<0>,
    %outer: !trait.claim<@Outer[i64]>,
    %claim: !trait.claim<@Trait[!trait.poly<0>, !trait.proj<@Outer[i64], "Item">]>) -> i64 {
  %x = arith.constant 1 : i64
  %eq = trait.allege !trait.proj<@Outer[i64], "Item"> = i64
  %px = trait.coerce %x : i64 to !trait.proj<@Outer[i64], "Item">
    via (%eq) : (!trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>)
  %result = trait.func.call @Trait.method(%claim, %t, %px) : (!trait.claim<@Trait[!trait.poly<0>, !trait.proj<@Outer[i64], "Item">]>, !trait.poly<0>, !trait.proj<@Outer[i64], "Item">) -> i64
  return %result : i64
}

trait.impl private @Outer_i64(%self: !trait.claim<@Outer[i64]>) bindings {Item = i64} {
}

trait.impl private @Trait_i64(%self_claim: !trait.claim<@Trait[i64, i64]>) members [@Trait.method -> @Trait_i64.method] {
}
func.func private @Trait_i64.method(%self_claim: !trait.claim<@Trait[i64, i64]>, %self: i64, %x: i64) -> i64 {
  return %x : i64
}

func.func @main() -> i64 {
  %outer = trait.witness @Outer_i64 for @Outer[i64]
  %trait = trait.witness @Trait_i64 for @Trait[i64, i64]
  %eq = trait.witness proj_resolve !trait.proj<@Outer[i64], "Item"> resolves i64 by @Outer_i64
    : !trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>
  %projected = trait.coerce %trait
    : !trait.claim<@Trait[i64, i64] by @Trait_i64>
    to !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">]>
    via (%eq) : (!trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>)
  %x = arith.constant 0 : i64
  %result = trait.func.call @callee(%x, %outer, %projected)
    : (i64, !trait.claim<@Outer[i64] by @Outer_i64>,
       !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">]>) -> i64
  return %result : i64
}

// CHECK-LABEL: func.func private @callee_
// CHECK: call @Trait_i64.method_{{h[0-9a-f]+}}
// CHECK-LABEL: func.func private @Trait_i64.method_{{h[0-9a-f]+}}
// CHECK: return %arg1 : i64
// CHECK-LABEL: func.func @main() -> i64
// CHECK: call @callee_
// CHECK: return {{.*}} : i64
