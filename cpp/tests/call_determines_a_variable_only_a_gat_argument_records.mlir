// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A callee variable stands in no position of the call but the associated-type
// argument of a projection, so the reading that pairs the declaration against
// the call's types fills it only after rebuilding the declaration at what it has
// read and reducing that projection: V := i1. The call lowers to a func.call to
// a fully monomorphic clone.
//
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// The clone is monomorphic: the GAT-determined parameter resolved to i1,
// and the call is a plain func.call to it.
// CHECK: func.func {{.*}}@foo_{{[a-z0-9]+}}(%{{.*}}: i64, %{{.*}}: i1,
// CHECK: call @foo_

trait.trait private @Trait[!trait.poly<79>] {
  trait.assoc_type @Assoc<[!trait.poly<80>]>
}
trait.impl private @Trait_impl for @Trait[i64] {
  trait.assoc_type @Assoc<[!trait.poly<177>]> = !trait.poly<177>
}
func.func private @foo(%arg0: !trait.poly<182>, %arg1: !trait.proj<@Trait[!trait.poly<182>], "Assoc", [!trait.poly<183>]>, %arg2: !trait.claim<@Trait[!trait.poly<182>]>) -> !trait.proj<@Trait[!trait.poly<182>], "Assoc", [!trait.poly<183>]> {
  return %arg1 : !trait.proj<@Trait[!trait.poly<182>], "Assoc", [!trait.poly<183>]>
}
func.func @main() -> i1 {
  %c0_i64 = arith.constant 0 : i64
  %true = arith.constant true
  %5 = trait.witness proj_resolve !trait.proj<@Trait[i64], "Assoc", [i1]> resolves i1 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1>
  %6 = trait.coerce %true : i1 to !trait.proj<@Trait[i64], "Assoc", [i1]> via (%5) : (!trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1>)
  %7 = trait.witness @Trait_impl for @Trait[i64]
  %8 = trait.func.call @foo(%c0_i64, %6, %7) : (i64, !trait.proj<@Trait[i64], "Assoc", [i1]>, !trait.claim<@Trait[i64] by @Trait_impl>) -> !trait.proj<@Trait[i64], "Assoc", [i1]>
  %9 = trait.witness proj_resolve !trait.proj<@Trait[i64], "Assoc", [i1]> resolves i1 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1>
  %10 = trait.coerce %8 : !trait.proj<@Trait[i64], "Assoc", [i1]> to i1 via (%9) : (!trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1>)
  return %10 : i1
}
