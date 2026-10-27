// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A binding cycle that grows rather than oscillates: @Grow[i32]'s Output
// resolves to a tuple that itself contains @Grow[i32]'s Output, so reading the
// answer reaches the projection still being read and the chain never ends. The
// stage resolves the projection where the call is lowered and reports the
// endless chain as its overflow, so the type never runs the process out of
// stack.

// CHECK: error: overflow evaluating the requirement '!trait.proj<@Grow[i32], "Output">': a chain of steps resolving its projections does not end within 128 steps

!T = !trait.poly<0>

trait.trait private @Grow(%self: !trait.claim<@Grow[!T]>) {
  trait.assoc_type @Output
}

trait.impl private @Grow_i32(%self: !trait.claim<@Grow[i32]>) {
  trait.assoc_type @Output = tuple<!trait.proj<@Grow[i32], "Output">, i32>
}

!W = !trait.poly<1>
trait.trait private @Wants(%self: !trait.claim<@Wants[!trait.poly<0>]>) { trait.method @m() -> i64 }

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
  %v = trait.method.call %w @m() : !trait.claim<@Wants[!trait.proj<@Grow[i32], "Output">] by @p>, () -> i64
  return %v : i64
}
