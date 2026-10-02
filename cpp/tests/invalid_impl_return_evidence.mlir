// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// An impl returns one evidence value per requirement of its trait, in order,
// and none of them may rest on the impl's own claim: a return one value short,
// one whose values stand in the wrong order, and one computing its evidence
// from %self are refused.

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> (!trait.claim<@B[!T]>, !trait.claim<@C[!T]>) {}
// expected-error @+1 {{returns 1 claims, and trait '@A' requires 2}}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %c: !trait.claim<@C[!T]>) {
  trait.return %b : !trait.claim<@B[!T]>
}

// -----

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> (!trait.claim<@B[!T]>, !trait.claim<@C[!T]>) {}
// expected-error @+1 {{returns '!trait.claim<@C[!trait.poly<0>]>' for requirement 0, which trait '@A' states as '!trait.claim<@B[!trait.poly<0>]>'}}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %c: !trait.claim<@C[!T]>) {
  trait.return %c, %b : !trait.claim<@C[!T]>, !trait.claim<@B[!T]>
}

// -----

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
// expected-error @+1 {{returns evidence for requirement 0 that rests on the impl's own application}}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>) {
  %b = trait.project %self[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
  trait.return %b : !trait.claim<@B[!T]>
}
