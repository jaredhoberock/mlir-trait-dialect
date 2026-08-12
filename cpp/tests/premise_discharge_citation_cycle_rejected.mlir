// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Discharge citations are checked over a finite list with cycle refusal: a
// citation may not discharge an obligation that is already on the active
// resolution stack. The premise cites @Sib_i64_cond (assumes @A[i64]). The
// citation for @A[i64] names @A_cond, which assumes @B[i64]; the citation for
// @B[i64] names @B_cond, which assumes @A[i64] again. Neither conditional impl
// has a base case, so following the citations re-enters @A[i64] under
// resolution -- a cycle that grounds nothing -- and the obligation stays
// undischarged. The impl is refused at birth rather than looping.

!S = !trait.poly<0>

trait.trait @A[!S] {}
trait.trait @B[!S] {}

trait.impl @A_cond for @A[i64] where [@B[i64]] {}
trait.impl @B_cond for @B[i64] where [@A[i64]] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64_cond for @Sib[i64] where [@A[i64]] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{cited impl '@Sib_i64_cond' has an undischarged assumption '!trait.claim<@A[i64]>'; the witness premises do not supply it}}
trait.impl @Host_i64 for @Host[i64]
    premises [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i32 by @Sib_i64_cond>]
    discharges [#trait<witness @A[i64] by @A_cond>,
                #trait<witness @B[i64] by @B_cond>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
