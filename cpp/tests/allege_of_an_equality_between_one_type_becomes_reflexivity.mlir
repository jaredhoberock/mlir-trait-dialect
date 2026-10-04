// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

// An allegation of an equality whose sides are one type spells no projection
// to resolve: it becomes the reflexive witness.

func.func private @need(!trait.claim<i64 = i64>)

func.func @main() {
  %e = trait.allege i64 = i64
  func.call @need(%e) : (!trait.claim<i64 = i64>) -> ()
  return
}

// ROUNDTRIP: trait.allege i64 = i64

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.allege
// CHECK: %[[REFL:.*]] = trait.witness refl : !trait.claim<i64 = i64>
// CHECK: call @need(%[[REFL]])
