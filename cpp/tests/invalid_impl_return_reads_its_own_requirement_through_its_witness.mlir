// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @A_i32 returns, for @A's requirement @B[i32], requirement 0 of its own
// application, named by a witness of itself rather than by %self: the evidence
// it returns for the requirement is that requirement, read back through the
// very return it stands in. It has no base case, and the impl is refused where
// it is declared.

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
// expected-error @below {{returns evidence for requirement 0 that projects the impl's own application}}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
