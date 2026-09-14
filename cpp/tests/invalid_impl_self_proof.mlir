// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics -split-input-file

trait.trait private @A[!trait.poly<0>] {}

trait.impl private @A_impl for @A[!trait.poly<1>] {}

trait.trait private @B[!trait.poly<2>] {}

trait.impl private @B_impl for @B[!trait.poly<3>] where [
  @A[!trait.poly<3>]
] {}

// expected-error @+1 {{'@A_impl' binds type parameters, states its own where clause, or implements a trait requiring an application, so it must be cited through a trait.proof}}
trait.proof private @B_impl_p proves @B_impl for @B[i8] given [@A_impl]
