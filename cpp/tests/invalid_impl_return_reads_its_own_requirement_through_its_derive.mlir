// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @A_any returns, for @A's requirement @B[T], requirement 0 of a derive of
// itself at its own application. A derive names no proof, but it states the
// impl it cites at the arguments it gives it, and the requirement it reads is
// the impl's own return there: the evidence rests on itself, and the impl is
// refused.

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
// expected-error @below {{returns evidence for requirement 0 that projects the impl's own application}}
trait.impl private @A_any(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>) {
  %a = trait.derive @A[!T] from @A_any[!trait.poly<0>] given(%b) : (!trait.claim<@B[!T]>)
  %r = trait.project %a[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
  trait.return %r : !trait.claim<@B[!T]>
}
