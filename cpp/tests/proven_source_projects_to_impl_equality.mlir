// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A proven claim's requirements continue past its trait's into the where
// entries of the impl its proof derives, so @T_impl's own equality entry stands
// at index 0 of a claim of @T[i64] by @T_p (the trait requires nothing), read
// off the derive's operand. An equality claim never carries a proof, so the
// result reads no provider.

// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Assoc(%self: !trait.claim<@Assoc[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @Assoc_i64(%self: !trait.claim<@Assoc[i64]>) { trait.assoc_type @Out = i32 }
trait.impl private @T_impl(%self: !trait.claim<@T[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) {}
trait.proof private @T_p {
  %p0 = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Out"> resolves i32 by @Assoc_i64
    : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  %d = trait.derive @T[i64] from @T_impl[i64] given(%p0) : (!trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>)
  trait.return %d : !trait.claim<@T[i64]>
}

// CHECK: trait.project %{{.*}}[0] : <@T[i64] by @T_p> -> <!trait.proj<@Assoc[i64], "Out"> = i32>
func.func @f(%s: !trait.claim<@T[i64] by @T_p>) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32> {
  %e = trait.project %s[0] : !trait.claim<@T[i64] by @T_p> -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  return %e : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
}
