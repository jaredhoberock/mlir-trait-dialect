// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The obligation @B[i32] carries is @A[i32]. A citation of an impl of an
// unrelated trait discharges nothing, however well that impl's header carries
// to its own claim.

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @C[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {}
trait.impl private @B_i32 for @B[i32] {}
trait.impl private @C_f32 for @C[f32] {}

// expected-error @below {{proof @C_f32 proves '!trait.claim<@C[f32]>', which does not discharge the obligation '!trait.claim<@A[i32]>'}}
trait.proof private @forged proves @B_i32 for @B[i32] given [@C_f32]
