// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A binding cycle that grows rather than oscillates: @Grow[i32]'s Output
// resolves to a tuple that itself contains @Grow[i32]'s Output, so each
// resolution pass nests the spelling one level deeper and it never settles. The
// fixed-point driver's rewrite budget bounds the growth and proof verification
// reports the nonconvergence cleanly, so the type never runs the process out of
// stack.

!T = !trait.poly<0>

trait.trait private @Grow[!T] {
  trait.assoc_type @Output
}

trait.impl private @Grow_i32 for @Grow[i32] {
  trait.assoc_type @Output = tuple<!trait.proj<@Grow[i32], "Output">, i32>
}

!W = !trait.poly<1>
trait.trait private @Wants[!W] {}

trait.impl private @Wants_impl for @Wants[!trait.proj<@Grow[i32], "Output">] {}

// expected-error @+1 {{projection normalization did not converge within 64 iterations}}
trait.proof private @p proves @Wants_impl for @Wants[!trait.proj<@Grow[i32], "Output">] given []
