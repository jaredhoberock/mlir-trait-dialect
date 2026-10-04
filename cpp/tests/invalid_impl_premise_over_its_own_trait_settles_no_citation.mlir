// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s -verify-diagnostics

// @A_gen's !T stands in no header position and two where-clause equalities
// spell it. Selection reads !T off the second, @B[!S]::X, whose type at i64
// is i1, as the impl's verifier counts it (rustc's E0207 skips a projection of
// the trait being implemented). The first projects the impl's own trait
// application, @A[!S]::Output, which names a type only through @A_gen's own
// binding of Output to !T: its evidence at i64 would be the citation of @A_gen
// it is a premise of, so no proof can supply it and the citation is refused,
// as rustc refuses a where clause that requires the impl's own application.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @B(%self: !trait.claim<@B[!S]>) {
  trait.assoc_type @X
}
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) {
  trait.assoc_type @X = i1
}

trait.trait private @A(%self: !trait.claim<@A[!S]>) {
  trait.assoc_type @Output
}
trait.impl private @A_gen(%self: !trait.claim<@A[!S]>, %b: !trait.claim<@B[!S]>, %output: !trait.claim<!trait.proj<@A[!S], "Output"> = !T>, %x: !trait.claim<!trait.proj<@B[!S], "X"> = !T>) {
  trait.assoc_type @Output = !T
}

func.func private @need(!trait.claim<@A[i64]>)

func.func @main() {
  // expected-error @below {{impl '@A_gen' applies where '!trait.claim<!trait.proj<@A[i64], "Output"> = i1>', which selection does not settle at '!trait.claim<@A[i64]>'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@A[i64]>' after instantiate-monomorphs}}
  %c = trait.allege @A[i64]
  func.call @need(%c) : (!trait.claim<@A[i64]>) -> ()
  return
}
