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

trait.trait private @X(!trait.claim<@X[!S]>)
trait.impl private @X_i64(%self: !trait.claim<@X[i64]>) {
}

trait.trait private @Sib(!trait.claim<@Sib[!S]>) assoc {Elem = 0}
trait.impl private @Sib_cond(%self: !trait.claim<@Sib[i64]>, %x: !trait.claim<@X[i64]>) bindings {Elem = i64} {
}

trait.trait private @T2(!trait.claim<@T2[!S]>) members [@T2.id]
func.func private @T2.id(!trait.claim<@T2[!S]>, !S) -> !S

// CHECK: trait.impl private @T2_i64(%self: !trait.claim<@T2[i64]>
trait.impl private @T2_i64(%self: !trait.claim<@T2[i64]>, %elem: !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>) members [@T2.id -> @T2_i64.id] {
}
func.func private @T2_i64.id(%self: !trait.claim<@T2[i64]>, %elem: !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>, %x: i64) -> i64 {
  return %x : i64
}
