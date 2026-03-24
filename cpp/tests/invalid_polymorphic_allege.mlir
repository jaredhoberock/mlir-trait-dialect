// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// Test that a polymorphic allege WITHOUT unsafe is rejected by the verifier.

!T = !trait.poly<0>

trait.trait @Marker [!T] {}

func.func @test_invalid(%x: i32) -> i32 {
  // CHECK: expected monomorphic claim
  %c = trait.allege @Marker[!T]
  return %x : i32
}
