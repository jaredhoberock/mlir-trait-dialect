// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_FREEZE_INSTANTIATION=1 not --crash mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// The instantiation driver still generates impls of its own, which is work its
// scheduler did not plan for: an impl built while the driver runs is a fact the
// run's earlier rewrites could not see. A freeze over that span says so where
// it happens, naming the claim selection asked for and the span whose contract
// the ask broke.
//
// @Gen[i64] has no impl, so resolving the projection over it reaches the
// generator arm. Without the freeze the ask is simply counted and the module
// compiles to the leftover-projection diagnostic the second run pins.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

func.func @wrap(%x: !T) -> !trait.proj<@Gen[!T], "A"> {
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Gen[!T], "A">
  return %r : !trait.proj<@Gen[!T], "A">
}

func.func @main() -> !trait.proj<@Gen[i64], "A"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "A">
  return %r : !trait.proj<@Gen[i64], "A">
}

// CHECK: impl generation is frozen for the instantiation driver, but impl selection demanded an impl of @Gen for !trait.claim<@Gen[i64]>
