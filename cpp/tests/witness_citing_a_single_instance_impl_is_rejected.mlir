// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A witness is verified by a RIGID head match, so a projection quantified over
// the host impl's parameter matches only a cited impl whose head is equally
// quantified. Here the host impl is generic over !S and the witness projection
// proj<@Sib[!S],"Elem"> cites the single-instance impl @Sib_i64: the match
// would have to bind !S to that one head, which would accept a generic impl on
// the strength of ONE instance. The head match refuses.

!S = !trait.poly<0>

trait.trait private @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64 for @Sib[i64] {
  trait.assoc_type @Elem = i32
}

trait.trait private @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{impl '@Sib_i64' at the witness's arguments is an impl for '!trait.claim<@Sib[i64]>', not for the projection's application '!trait.claim<@Sib[!trait.poly<0>]>'}}
trait.impl private @Host_T for @Host[!S]
    witnesses [#trait<witness !trait.proj<@Sib[!S], "Elem"> = i32 by @Sib_i64>] {
  func.func @make(%x: !S) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
