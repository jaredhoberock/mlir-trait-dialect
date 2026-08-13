// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A discharge citation supplies an obligation only when its NAMED impl genuinely
// discharges it -- the named impl's own assumptions must in turn be discharged.
// The premise cites @Sib_i64_cond (assumes @Y[i64]); the citation names @Y_cond
// as the discharger of @Y[i64], but @Y_cond itself assumes @Z[i64], and no
// citation and no where entry supplies @Z[i64]. Verification follows the citation
// to @Y_cond, finds its assumption @Z[i64] undischarged, and refuses -- the
// obligation @Y[i64] therefore stays undischarged and the impl is refused at
// impl verification.

!S = !trait.poly<0>

trait.trait @Y[!S] {}
trait.trait @Z[!S] {}

trait.impl @Y_cond for @Y[i64] where [@Z[i64]] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64_cond for @Sib[i64] where [@Y[i64]] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{cited impl '@Sib_i64_cond' has an undischarged assumption '!trait.claim<@Y[i64]>'; the witness premises do not supply it}}
trait.impl @Host_i64 for @Host[i64]
    witnesses [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i32 by @Sib_i64_cond>,
               #trait<witness @Y[i64] by @Y_cond>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
