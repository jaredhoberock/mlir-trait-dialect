// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proj_resolve witness is verified by a RIGID head match, so a projection
// quantified over the host impl's parameter matches only a cited impl whose head
// is equally quantified. Here the host impl is generic over !S and the witness
// in its method resolves proj<@Sib[!S],"Elem"> by the single-instance impl
// @Sib_i64: the match would have to bind !S to that one head, which would
// accept a generic impl on the strength of ONE instance. The head match
// refuses.

!S = !trait.poly<0>

trait.trait private @Sib(%self: !trait.claim<@Sib[!S]>) {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64(%self: !trait.claim<@Sib[i64]>) {
  trait.assoc_type @Elem = i32
}

trait.trait private @Host(%self: !trait.claim<@Host[!S]>) {
  trait.method @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

trait.impl private @Host_T(%self: !trait.claim<@Host[!S]>) {
  trait.method @make(%x: !S) -> !trait.proj<@Sib[!S], "Elem"> {
    %r = ub.poison : i32
    // expected-error @below {{impl '@Sib_i64' at the arguments the citation gives it proves '!trait.claim<@Sib[i64]>', not '!trait.claim<@Sib[!trait.poly<0>]>'}}
    %e = trait.witness proj_resolve !trait.proj<@Sib[!S], "Elem"> resolves i32 by @Sib_i64
      : !trait.claim<!trait.proj<@Sib[!S], "Elem"> = i32>
    %c = trait.coerce %r : i32 to !trait.proj<@Sib[!S], "Elem"> via (%e)
      : (!trait.claim<!trait.proj<@Sib[!S], "Elem"> = i32>)
    trait.return %c : !trait.proj<@Sib[!S], "Elem">
  }
}
