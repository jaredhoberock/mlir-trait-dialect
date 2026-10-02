// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' 2>&1 | FileCheck %s

// @T2_i64 assumes proj<@Sib[i64],"Elem"> = i1 and declares a witness -- citing
// the conditional @Sib_cond, its @X[i64] assumption supplied by a discharge
// citation -- that resolves the projection to i64. The assumption is a premise,
// so the impl itself verifies; selection is what discharges it, and there the
// projection reduces through @Sib_cond to the ground i64, which is not i1. The
// demand @T2[i64] is refused.

// CHECK: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@T2[i64]>'

!S = !trait.poly<0>

trait.trait private @X(%self: !trait.claim<@X[!S]>) {}
trait.impl private @X_i64(%self: !trait.claim<@X[i64]>) {}

trait.trait private @Sib(%self: !trait.claim<@Sib[!S]>) {
  trait.assoc_type @Elem
}
trait.impl private @Sib_cond(%self: !trait.claim<@Sib[i64]>, %x: !trait.claim<@X[i64]>) {
  trait.assoc_type @Elem = i64
}

trait.trait private @T2(%self: !trait.claim<@T2[!S]>) {
  trait.method @id(!S) -> !S
}

trait.impl private @T2_i64(%self: !trait.claim<@T2[i64]>, %elem: !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i1>) {
  trait.method @id(%x: i64) -> i64 {
    trait.return %x : i64
  }
}

func.func private @needs(!trait.claim<@T2[i64]>)

func.func @main() {
  %c = trait.allege @T2[i64]
  trait.func.call @needs(%c) : (!trait.claim<@T2[i64]>) -> ()
  return
}
