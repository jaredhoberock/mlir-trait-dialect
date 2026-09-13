// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// @Has declares one requirement and the source claim is unproven, so it names
// no impl whose assumptions would continue the list. Index 1 reaches past the
// end, and the refusal names how many requirements the claim carries.

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

!S = !trait.poly<0>

trait.trait private @A[!trait.poly<1>] {}
trait.trait private @Has[!S] where [@A[!S]] {}

func.func @f(%p: !trait.claim<@Has[i32]>) -> !trait.claim<@A[i32]> {
  // CHECK: requirement index 1 is out of range
  // CHECK-SAME: has 1 requirements
  %a = trait.project %p[1] : !trait.claim<@Has[i32]> -> !trait.claim<@A[i32]>
  return %a : !trait.claim<@A[i32]>
}
