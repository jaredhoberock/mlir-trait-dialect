// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proof citing itself stands for its own claim, which discharges an
// obligation only when the obligation is that claim. Here the obligation is
// @A[i32] and the proof proves @B[i32].

trait.trait private @A[!trait.poly<0>] {}

trait.trait private @B[!trait.poly<1>] where [@A[!trait.poly<1>]] {}

trait.impl private @B_impl for @B[i32] {}

trait.impl private @A_impl for @A[i32] {}

// expected-error @+1 {{proof @self_proof proves '!trait.claim<@B[i32]>', which does not discharge the obligation '!trait.claim<@A[i32]>'}}
trait.proof private @self_proof proves @B_impl for @B[i32] given [@self_proof]
