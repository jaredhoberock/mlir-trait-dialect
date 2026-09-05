// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proof must not reference itself as a sub-proof.

trait.trait private @A[!trait.poly<0>] {}

trait.trait private @B[!trait.poly<1>] where [@A[!trait.poly<1>]] {}

trait.impl private @B_impl for @B[i32] {}

trait.impl private @A_impl for @A[i32] {}

// expected-error @+1 {{sub-proof '@self_proof' must not reference the proof itself (proves @B but obligation requires @A)}}
trait.proof private @self_proof proves @B_impl for @B[i32] given [@self_proof]
