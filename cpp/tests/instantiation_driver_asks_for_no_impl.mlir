// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_FREEZE_INSTANTIATION=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// The instantiation driver reads the facts the steps before it settled and
// asks impl selection for nothing, so a freeze standing over that span is
// silent and the two runs below agree.
//
// @Gen[i64] has no impl and no generator supplies one, so the projection over
// it is a demand the driver declines and a later round collects. That round
// puts it to selection, where generating is legal; selection still finds
// nothing, and the projection reaches the end of the stage spelled as written.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

func.func @wrap(%x: !T) -> !trait.proj<@Gen[!T], "A"> {
  %r = ub.poison : !trait.proj<@Gen[!T], "A">
  return %r : !trait.proj<@Gen[!T], "A">
}

func.func @main() -> !trait.proj<@Gen[i64], "A"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "A">
  return %r : !trait.proj<@Gen[i64], "A">
}
