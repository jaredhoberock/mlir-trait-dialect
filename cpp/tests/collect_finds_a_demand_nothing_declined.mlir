// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The initial demand collection reaches a monomorphic claim in an otherwise
// polymorphic function signature. The unused template is erased successfully.

!T = !trait.poly<0>

trait.trait private @P[!T] {}

trait.impl private @P_i64 for @P[i64] {}

func.func nested @holds(%c: !trait.claim<@P[i64]>, %x: !T) -> !T {
  return %x : !T
}

// CHECK: module {
// CHECK-NEXT: }
