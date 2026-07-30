// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s

// Two impls bind @Gen[i64] and both their assumption lists are empty, so impl
// selection refuses the application for having two satisfiable candidates. That
// is the refusal no later round overturns -- candidates are only ever appended,
// so a partition that already had two keeps at least two -- which is why the
// round keeps it where it forgets the other kind, and why the demand it refused
// leaves the drain rather than waiting for facts that cannot help it.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_wide for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.impl @Gen_narrow for @Gen[i64] {
  trait.assoc_type @A = i16
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

// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=1 no-candidate-impl=0 multiple-candidate-impls=1
// CHECK-SAME: served=0 declined=1 deferred=0
// CHECK-SAME: refusals-forgotten=0 refusals-kept=1
// CHECK: trait-stage-record digest value={{.*}} refusals-no-candidate=0 refusals-ambiguous=1
