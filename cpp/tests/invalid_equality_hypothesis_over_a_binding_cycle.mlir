// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// An equality hypothesis over two projections whose bindings name each other
// verifies as a hypothesis, and the coercion it carries is refused where the
// stage resolves the projections: each binding grows by one tuple level per
// step, so selection stops at its depth limit rather than looping.

trait.trait private @Grow(%s: !trait.claim<@Grow[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.trait private @Loop(%s: !trait.claim<@Loop[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Grow_any(%s: !trait.claim<@Grow[!trait.poly<0>]>) { trait.assoc_type @Out = tuple<!trait.proj<@Loop[!trait.poly<0>], "Out">> }
trait.impl private @Loop_any(%s: !trait.claim<@Loop[!trait.poly<0>]>) { trait.assoc_type @Out = tuple<!trait.proj<@Grow[!trait.poly<0>], "Out">> }
func.func @test(%x: !trait.proj<@Grow[i32], "Out">, %eq: !trait.claim<!trait.proj<@Grow[i32], "Out"> = !trait.proj<@Loop[i32], "Out">>) -> !trait.proj<@Loop[i32], "Out"> {
  // expected-error @below {{overflow evaluating the requirement '!trait.proj<@Loop[i32], "Out">': 128 projection steps stand on the chain}}
  %r = trait.coerce %x : !trait.proj<@Grow[i32], "Out"> to !trait.proj<@Loop[i32], "Out"> via (%eq) : (!trait.claim<!trait.proj<@Grow[i32], "Out"> = !trait.proj<@Loop[i32], "Out">>)
  return %r : !trait.proj<@Loop[i32], "Out">
}
