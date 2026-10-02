// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s --pass-pipeline='builtin.module(trait.impl(trait.method(canonicalize)))' 2>&1 | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline='builtin.module(trait.impl(canonicalize))' | FileCheck %s --check-prefix=IMPL

// A method is not isolated from above, so no pass may anchor on it: the pass
// manager refuses a pipeline nested on one. A pipeline nested on the impl, which
// is isolated, runs over the impl's methods.

// CHECK: 'trait.method' op trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'

// IMPL-LABEL: trait.impl private @Tr_i64
// IMPL-NEXT:    trait.method @a
// IMPL-NEXT:      arith.constant 8 : i64

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) {
  trait.method @a(!T) -> i64
}

trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @a(%x: i64) -> i64 {
    %seven = arith.constant 7 : i64
    %one = arith.constant 1 : i64
    %eight = arith.addi %seven, %one : i64
    trait.return %eight : i64
  }
}
