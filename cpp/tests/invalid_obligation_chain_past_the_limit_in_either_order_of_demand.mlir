// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | not mlir-opt -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @W0[i32] stands on a chain of one hundred and twelve selections, and
// @V0[i32] on thirty-one more above it, past the obligation limit. @main's
// parameters demand both, in either order, and whichever is asked first,
// @V0[i32] overflows: a selection answered before stands as high above the
// chain reading it as its derivation did. The overflow is the one diagnostic,
// and @W0[i32] alone is within the limit.

// CHECK: error: overflow evaluating the requirement
// CHECK-NOT: no impl with satisfiable assumptions
// CHECK: error: overflow evaluating the requirement
// CHECK-NOT: no impl with satisfiable assumptions

!T = !trait.poly<0>
// REPEAT 0 111: trait.trait private @W{k}(%s: !trait.claim<@W{k}[!T]>) {}
// REPEAT 0 110: trait.impl private @W{k}_i32(%s: !trait.claim<@W{k}[i32]>, %n: !trait.claim<@W{k+1}[i32]>) {}
trait.impl private @W111_i32(%s: !trait.claim<@W111[i32]>) {}
// REPEAT 0 30: trait.trait private @V{k}(%s: !trait.claim<@V{k}[!T]>) {}
// REPEAT 0 29: trait.impl private @V{k}_i32(%s: !trait.claim<@V{k}[i32]>, %n: !trait.claim<@V{k+1}[i32]>) {}
trait.impl private @V30_i32(%s: !trait.claim<@V30[i32]>, %w: !trait.claim<@W0[i32]>) {}
func.func @main(%w: !trait.claim<@W0[i32]>, %v: !trait.claim<@V0[i32]>) {
  return
}

// -----

!T = !trait.poly<0>
// REPEAT 0 111: trait.trait private @W{k}(%s: !trait.claim<@W{k}[!T]>) {}
// REPEAT 0 110: trait.impl private @W{k}_i32(%s: !trait.claim<@W{k}[i32]>, %n: !trait.claim<@W{k+1}[i32]>) {}
trait.impl private @W111_i32(%s: !trait.claim<@W111[i32]>) {}
// REPEAT 0 30: trait.trait private @V{k}(%s: !trait.claim<@V{k}[!T]>) {}
// REPEAT 0 29: trait.impl private @V{k}_i32(%s: !trait.claim<@V{k}[i32]>, %n: !trait.claim<@V{k+1}[i32]>) {}
trait.impl private @V30_i32(%s: !trait.claim<@V30[i32]>, %w: !trait.claim<@W0[i32]>) {}
func.func @main(%v: !trait.claim<@V0[i32]>, %w: !trait.claim<@W0[i32]>) {
  return
}
