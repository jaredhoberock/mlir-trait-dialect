// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -stats -verify-diagnostics 2>&1 | FileCheck %s

// The census is one of two channels. This is the other: statistics that tick
// whether or not the census is switched on, which is how a demand raised
// outside any stage -- a verifier's -- is counted at all. The row pins that the
// swallowed failures of the read the instantiation driver holds reach that
// channel.

trait.trait @T[!trait.poly<0>] {
  trait.assoc_type @A
}

func.func @main() -> !trait.proj<@T[i64], "A"> {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@T[i64], "A">
  return %r : !trait.proj<@T[i64], "A">
}

// CHECK: 4 trait-demand
// CHECK-SAME: demands a read of the recorded facts had no answer for
