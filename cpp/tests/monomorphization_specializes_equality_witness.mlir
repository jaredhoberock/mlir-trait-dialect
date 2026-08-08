// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A generic body carries an equality witness whose endpoints mention the type
// parameter, and a coerce consuming it. Monomorphizing the caller runs the
// clone rule over that witness. The rule specializes the endpoints by pure
// substitution -- @Trait[!poly<0>]::Output = !poly<0> becomes
// @Trait[i64]::Output = i64, a single substitution instance of the frozen
// certificate the witness check accepts -- and the concrete projection then
// resolves and the coerce discharges, leaving the identity the source denotes.
// The whole path clones, checks, and lowers with no leftover claim.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

trait.proof @Trait_i64_p proves @Trait_impl for @Trait[i64] given []

func.func @gen(%p: !trait.proj<@Trait[!S], "Output">, %c: !trait.claim<@Trait[!S]>) -> !S {
  %e = trait.witness proj_resolve !trait.proj<@Trait[!S], "Output"> resolves !S by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S>
  %v = trait.coerce %p : !trait.proj<@Trait[!S], "Output"> to !S via (%e) : (!trait.claim<!trait.proj<@Trait[!S], "Output"> = !S>)
  return %v : !S
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: trait.witness
// CHECK-NOT: trait.coerce
// CHECK: call @gen
func.func @caller() -> i64 {
  %x = arith.constant 7 : i64
  %w = trait.witness @Trait_i64_p for @Trait[i64]
  %pw = trait.proj.cast %x, %w : i64 to !trait.proj<@Trait[i64], "Output"> by !trait.claim<@Trait[i64] by @Trait_i64_p>
  %r = trait.func.call @gen(%pw, %w) : (!trait.proj<@Trait[i64], "Output">, !trait.claim<@Trait[i64] by @Trait_i64_p>) -> i64
  return %r : i64
}
