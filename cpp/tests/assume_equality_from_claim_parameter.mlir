// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An equality-arm trait.assume anchored on a function equality-claim PARAMETER.
// The enclosing function carries a !trait.claim<!S = i32> parameter, so assuming
// !S = i32 is an axiom of this scope -- the parameter anchor of the equality arm,
// symmetric to the application arm's claim-parameter anchor. The assumed equality
// justifies a trait.coerce rewriting the ground value to the parameter type.

// CHECK-LABEL: func.func @gen
// CHECK: trait.assume !trait.poly<0> = i32

!S = !trait.poly<0>

func.func @gen(%c: !trait.claim<!S = i32>, %x: i32) -> !S {
  %e = trait.assume !S = i32
  %v = trait.coerce %x : i32 to !S via (%e) : (!trait.claim<!S = i32>)
  return %v : !S
}
