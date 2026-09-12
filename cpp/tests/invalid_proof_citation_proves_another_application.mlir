// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A citation discharges an obligation only when the claim it stands for is that
// obligation. @B[i32] requires @A[i32]; the proof cites @A_i64, an impl of the
// same trait at another argument. Reading the cited impl's own header against
// its own claim says nothing about the obligation, so the two must be compared.

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {}
trait.impl private @B_i32 for @B[i32] {}
trait.impl private @A_i64 for @A[i64] {}

// expected-error @below {{proof @A_i64 proves '!trait.claim<@A[i64]>', which does not discharge the obligation '!trait.claim<@A[i32]>'}}
trait.proof private @forged proves @B_i32 for @B[i32] given [@A_i64]
