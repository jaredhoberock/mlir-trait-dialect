// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWERED

// A ground projection one impl resolves stands only in the signature of a
// bodiless declaration nothing calls: no value carries it, but the type its
// function-type attribute carries is a standing site, and instantiation, which
// resolves a projection wherever an op spells one, owes the work there. The
// module is therefore pending, and the lowering driver runs instantiation
// before erasure, which resolves the projection rather than meet one it does
// not resolve; the collector then takes the declaration nothing names.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

// CHECK: pending-expansion=true
// LOWERED-NOT: trait.
// LOWERED: func.func @main() -> i64
// LOWERED-NOT: trait.
func.func private @decl(!trait.proj<@A[i64], "Item">) -> i64

func.func @main() -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}
