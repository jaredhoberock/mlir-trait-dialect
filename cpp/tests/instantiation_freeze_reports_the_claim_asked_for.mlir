// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not --crash mlir-opt %s -pass-pipeline='builtin.module(ask-impl-selection-during-instantiation-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A freeze stands over the instantiation driver on every compile, and the
// driver's patterns read what the steps before them recorded rather than making
// impl selection run, so nothing in a compilation ever asks it anything. This
// row is the ask: the pass on the first line adds a pattern that puts the claim
// @declares spells to impl selection from inside that driver, which is exactly
// what the freeze forbids. Selection finds no impl of @Gen, asks the generators
// for one, and the freeze names the claim and the span whose contract the ask
// broke.
//
// @declares carries no body on purpose. A round collects the claims the module
// spells in a result type or a block argument, and a function that only declares
// itself has neither -- its claim lives in the function type alone, which is
// where the pattern reads it. So this is an ask about an application no round
// ever put to selection, which is the only kind that can still reach a generator
// from under the driver.
//
// The second line is the same module through the compiler's own pipeline, where
// no pattern asks, so the freeze is silent and the stage completes.

!T = !trait.poly<0>

trait.trait @Gen[!T] {}

func.func private @declares(!trait.claim<@Gen[i64]>, !T) -> !T

func.func @main() -> i64 {
  %x = arith.constant 1 : i64
  return %x : i64
}

// CHECK: impl generation is frozen for the instantiation driver, but impl selection demanded an impl of @Gen for !trait.claim<@Gen[i64]>
