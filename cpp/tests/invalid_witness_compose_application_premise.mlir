// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A composition witness composes equality claims. An application-arm claim is
// not equality evidence and carries no endpoints for the closure to seed, so
// passing one as a premise is refused at the arm.

trait.trait @T[!trait.poly<0>] {}

func.func @application_premise(%p: !trait.claim<@T[i32]>) -> !trait.claim<i32 = i32> {
  // expected-error @+1 {{a composition witness premise must be an equality claim}}
  %c = trait.witness compose(%p) : (!trait.claim<@T[i32]>) : !trait.claim<i32 = i32>
  return %c : !trait.claim<i32 = i32>
}
