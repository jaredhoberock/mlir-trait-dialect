// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The accept companion to impl_own_false_equality_resolved_by_premise_refuses:
// the premise-resolved own equality is accepted when the ground value MATCHES.
// @T2_i64 assumes proj<@Sib[i64],"Elem"> = i64, and it declares a premise --
// citing the conditional @Sib_cond, its @X[i64] assumption supplied by a
// discharge citation -- that resolves the projection to i64. The own-equality impl-verification
// check replays the premise as the modulus it reduces the equality through,
// reduces the endpoint to the ground value i64, finds the equality holds, and
// the impl verifies. Only the ground MISMATCH refuses; a satisfied ground
// equality must not be over-refused.

!S = !trait.poly<0>

trait.trait @X[!S] {}
trait.impl @X_i64 for @X[i64] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}
trait.impl @Sib_cond for @Sib[i64] where [@X[i64]] {
  trait.assoc_type @Elem = i64
}

trait.trait @T2[!S] {
  func.func private @id(!S) -> !S
}

// CHECK: trait.impl @T2_i64 for @T2[i64]where [!trait.proj<@Sib[i64], "Elem"> = i64]
trait.impl @T2_i64 for @T2[i64] where [!trait.proj<@Sib[i64], "Elem"> = i64]
    witnesses [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i64 by @Sib_cond>,
               #trait<witness @X[i64] by @X_i64>] {
  func.func @id(%x: i64) -> i64 {
    return %x : i64
  }
}
