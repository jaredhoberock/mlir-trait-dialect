// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A premise is supplied only by evidence that holds on its own -- the impl it
// names must in turn have its where clause supplied. The proj_resolve witness
// cites @Sib_i64_cond (which takes @Y[i64]) and supplies, by position, a
// witness naming @Y_cond as the evidence for @Y[i64]; but @Y_cond itself takes
// @Z[i64], which no witness supplies: a conditional impl is cited through a
// proof that derives it from its premises, never named bare. The witness of
// @Y_cond is refused, so the premise it would supply does not exist.

!S = !trait.poly<0>

trait.trait private @Y(%self: !trait.claim<@Y[!S]>) {}
trait.trait private @Z(%self: !trait.claim<@Z[!S]>) {}

trait.impl private @Y_cond(%self: !trait.claim<@Y[i64]>, %z: !trait.claim<@Z[i64]>) {}

trait.trait private @Sib(%self: !trait.claim<@Sib[!S]>) {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64_cond(%self: !trait.claim<@Sib[i64]>, %y: !trait.claim<@Y[i64]>) {
  trait.assoc_type @Elem = i32
}

trait.trait private @Host(%self: !trait.claim<@Host[!S]>) {
  trait.method @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) {
  trait.method @make(%x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
    %r = ub.poison : i32
    // expected-error @below {{impl '@Y_cond' binds type parameters or has a where clause, so it must be cited through a trait.proof}}
    %y = trait.witness @Y_cond for @Y[i64]
    %e = trait.witness proj_resolve !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond
      given(%y) : (!trait.claim<@Y[i64] by @Y_cond>)
      : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i32>
    %c = trait.coerce %r : i32 to !trait.proj<@Sib[i64], "Elem"> via (%e)
      : (!trait.claim<!trait.proj<@Sib[i64], "Elem"> = i32>)
    trait.return %c : !trait.proj<@Sib[i64], "Elem">
  }
}
