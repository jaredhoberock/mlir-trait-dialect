// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait, an impl and a proof may be written public: a module exports its
// traits and their impls as public symbols. Their verifiers judge their
// structure alone; a template must be private only by the time erase runs.

// CHECK: trait.trait @Show(
trait.trait @Show(%self: !trait.claim<@Show[!trait.poly<0>]>) {
}

// CHECK: trait.impl @Show_i32(
trait.impl @Show_i32(%self: !trait.claim<@Show[i32]>) {
}

// CHECK: trait.proof @Show_i32_p
trait.proof @Show_i32_p {
  %d = trait.derive @Show[i32] from @Show_i32 given()
  trait.return %d : !trait.claim<@Show[i32]>
}
