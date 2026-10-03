// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | not mlir-opt -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A finite chain of proofs one hundred and twenty-nine deep, the shallowest
// that stands past the obligation limit: @p0 derives @P0[i32] over @p1's
// claim, @p1 over @p2's, and so on to @p128, which derives @P128[i32] from its
// impl alone. The derivation is refused before any instance is cut. Here the
// proofs stand from the top of the chain down.

// CHECK: error: overflow evaluating the requirement {{.*}}@P128[i32]{{.*}}: 128 obligations stand on the chain that reaches it

// REPEAT 0 128: trait.trait private @P{k}(%self: !trait.claim<@P{k}[!trait.poly<0>]>) {}
// REPEAT 0 127: trait.impl private @P{k}_i32(%self: !trait.claim<@P{k}[i32]>, %n: !trait.claim<@P{k+1}[i32]>) {}
trait.impl private @P128_i32(%self: !trait.claim<@P128[i32]>) {}
// REPEAT 0 127: trait.proof private @p{k} { %n = trait.witness @p{k+1} for @P{k+1}[i32] %d = trait.derive @P{k}[i32] from @P{k}_i32 given(%n) : (!trait.claim<@P{k+1}[i32] by @p{k+1}>) trait.return %d : !trait.claim<@P{k}[i32]> }
trait.proof private @p128 {
  %d = trait.derive @P128[i32] from @P128_i32 given()
  trait.return %d : !trait.claim<@P128[i32]>
}

// -----

// The same chain with its proofs standing from the bottom up. The verdict
// depends on no order proofs stand in.

// CHECK: error: overflow evaluating the requirement {{.*}}@P128[i32]{{.*}}: 128 obligations stand on the chain that reaches it

// REPEAT 0 128: trait.trait private @P{k}(%self: !trait.claim<@P{k}[!trait.poly<0>]>) {}
// REPEAT 0 127: trait.impl private @P{k}_i32(%self: !trait.claim<@P{k}[i32]>, %n: !trait.claim<@P{k+1}[i32]>) {}
trait.impl private @P128_i32(%self: !trait.claim<@P128[i32]>) {}
trait.proof private @p128 {
  %d = trait.derive @P128[i32] from @P128_i32 given()
  trait.return %d : !trait.claim<@P128[i32]>
}
// REPEAT 127 0: trait.proof private @p{k} { %n = trait.witness @p{k+1} for @P{k+1}[i32] %d = trait.derive @P{k}[i32] from @P{k}_i32 given(%n) : (!trait.claim<@P{k+1}[i32] by @p{k+1}>) trait.return %d : !trait.claim<@P{k}[i32]> }
