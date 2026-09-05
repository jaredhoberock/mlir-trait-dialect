// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A trait.project from a PROVEN source claim to one of the impl's equality
// where-clauses verifies. An equality claim never carries a proof, so parity is
// exempt for it; getProjections offers the impl's assumed equalities as
// candidates for a proven source, specialized through the self claim.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Assoc[!trait.poly<0>] { trait.assoc_type @Out }
trait.trait private @T[!trait.poly<1>] {}
trait.impl private @Assoc_i64 for @Assoc[i64] { trait.assoc_type @Out = i32 }
trait.impl private @T_impl for @T[!trait.poly<2>] where [!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32] {}
trait.proof private @T_p proves @T_impl for @T[i64] given []

// CHECK: trait.project %{{.*}} to !trait.proj<@Assoc[i64], "Out"> = i32
func.func @f(%s: !trait.claim<@T[i64] by @T_p>) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32> {
  %e = trait.project %s : @T[i64] by @T_p to !trait.proj<@Assoc[i64], "Out"> = i32
  return %e : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
}
