// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A binding cycle that grows rather than oscillates: @Grow[i32]'s Output
// resolves to a tuple that itself contains @Grow[i32]'s Output, so each
// resolution pass nests the spelling one level deeper and it never settles. The
// stage resolves the projection where the call is lowered; the fixed-point
// driver's rewrite budget bounds the growth and the nonconvergence is reported
// as the stage's overflow, so the type never runs the process out of stack.

// CHECK: error: overflow evaluating the requirement '!trait.proj<@Grow[i32], "Output">': 128 projection steps stand on the chain

!T = !trait.poly<0>

trait.trait private @Grow(%self: !trait.claim<@Grow[!T]>) {
  trait.assoc_type @Output
}

trait.impl private @Grow_i32(%self: !trait.claim<@Grow[i32]>) {
  trait.assoc_type @Output = tuple<!trait.proj<@Grow[i32], "Output">, i32>
}

!W = !trait.poly<1>
trait.trait private @Wants(%self: !trait.claim<@Wants[!W]>) { trait.method @m() -> i64 }

trait.impl private @Wants_impl(%self: !trait.claim<@Wants[!trait.proj<@Grow[i32], "Output">]>) {
  trait.method @m() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

trait.proof private @p {
  %d = trait.derive @Wants[!trait.proj<@Grow[i32], "Output">] from @Wants_impl given()
  trait.return %d : !trait.claim<@Wants[!trait.proj<@Grow[i32], "Output">]>
}

func.func @main() -> i64 {
  %w = trait.witness @p for @Wants[!trait.proj<@Grow[i32], "Output">]
  %v = trait.method.call %w @Wants[!trait.proj<@Grow[i32], "Output">]::@m() : () -> i64 by @p
  return %v : i64
}
