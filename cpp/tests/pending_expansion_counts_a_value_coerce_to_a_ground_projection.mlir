// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWERED

// A value respelled to a ground projection one impl resolves, then back. Every
// use of the first coerce is an op of this dialect, so it reads through, but
// what reading through exempts is an application claim, which carries its
// input's proof; the projection the coerce produces is a standing site
// instantiation owes, as erasure's entry judges it. The module is therefore
// pending instantiation, the lowering driver runs instantiate before erasure,
// and the function lowers to its argument rather than being refused at
// erasure's entry for a projection instantiation would have resolved.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

// CHECK: pending-expansion=true
// LOWERED-LABEL: func.func @use(%arg0: i64) -> i64
// LOWERED-NEXT: return %arg0 : i64
func.func @use(%x: i64) -> i64 {
  %w = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64] : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %p = trait.coerce %x : i64 to !trait.proj<@A[i64], "Item"> via (%w) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>)
  %y = trait.coerce %p : !trait.proj<@A[i64], "Item"> to i64 via (%w) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>)
  return %y : i64
}
