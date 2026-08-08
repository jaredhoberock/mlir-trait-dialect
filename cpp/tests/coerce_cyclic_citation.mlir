// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A coerce may cite a cyclic equality: nothing requires its equality operands to
// come from trait.witness, so a block argument of type claim<i1 = tuple<i1>> is
// spellable. Rebuilding every parent through its constructor to a fixed point
// would diverge on the cycle -- tuple<i1> rebuilds to tuple<tuple<i1>>, then
// deeper without end -- so the closure mints a rebuilt parent only when its
// constructor actually normalized and drops a free re-application. The verifier
// terminates and still decides the directly cited equality.
// CHECK-LABEL: func.func @cyclic
// CHECK: trait.coerce %arg0 : i1 to tuple<i1> via (%arg1) : (!trait.claim<i1 = tuple<i1>>)
func.func @cyclic(%x: i1, %e: !trait.claim<i1 = tuple<i1>>) -> tuple<i1> {
  %y = trait.coerce %x : i1 to tuple<i1> via (%e) : (!trait.claim<i1 = tuple<i1>>)
  return %y : tuple<i1>
}

// Entailment through the cycle: i64 = tuple<i64> entails i64 = tuple<tuple<i64>>
// by one congruence step over the existing terms, so pure closure decides it
// with no rebuild at all.
// CHECK-LABEL: func.func @cyclic_depth
// CHECK: trait.coerce %arg0 : i64 to tuple<tuple<i64>> via (%arg1) : (!trait.claim<i64 = tuple<i64>>)
func.func @cyclic_depth(%x: i64, %e: !trait.claim<i64 = tuple<i64>>) -> tuple<tuple<i64>> {
  %y = trait.coerce %x : i64 to tuple<tuple<i64>> via (%e) : (!trait.claim<i64 = tuple<i64>>)
  return %y : tuple<tuple<i64>>
}
