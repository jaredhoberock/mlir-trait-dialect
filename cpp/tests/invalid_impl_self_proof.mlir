// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics -split-input-file

trait.trait @A[!trait.poly<0>] {}

trait.impl @A_impl for @A[!trait.poly<1>] {}

trait.trait @B[!trait.poly<2>] {}

trait.impl @B_impl for @B[!trait.poly<3>] where [
  @A[!trait.poly<3>]
] {}

// expected-error @+1 {{'@A_impl' is polymorphic (has type parameters) or has obligations (trait requirements or impl assumptions) and must be proven with a trait.proof}}
trait.proof @B_impl_p proves @B_impl for @B[i8] given [@A_impl]
