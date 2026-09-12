// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' 2>&1 | FileCheck %s

// An impl's own where-clause equality is a premise: it restricts when the impl
// applies and is never an obligation the impl owes, so @FoldFn_bad verifies
// even though its own binding makes the premise false. What the premise decides
// is selection: discharging it for the demand @FoldFn[i32] reduces
// @FoldFn[i32]::Output through the candidate's own binding to i64, which is not
// i32, so the candidate is refused and the demand has no impl.

// CHECK: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@FoldFn[i32]>'

!S = !trait.poly<0>

trait.trait private @FoldFn[!S] {
  trait.assoc_type @Output
}

trait.impl private @FoldFn_bad for @FoldFn[i32] where [!trait.proj<@FoldFn[i32], "Output"> = i32] {
  trait.assoc_type @Output = i64
}

func.func private @needs(!trait.claim<@FoldFn[i32]>)

func.func @main() {
  %c = trait.allege @FoldFn[i32]
  trait.func.call @needs(%c) : (!trait.claim<@FoldFn[i32]>) -> ()
  return
}
