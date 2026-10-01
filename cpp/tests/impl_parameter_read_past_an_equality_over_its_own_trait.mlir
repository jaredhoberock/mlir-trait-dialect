// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

// @A_gen's !T stands in no header position and two where-clause equalities
// spell it. The first projects the impl's own trait application, @A[!S]::Output,
// which names a type only through @A_gen's own binding of Output to !T; reading
// it to determine !T would select @A_gen to read @A_gen's arguments again. The
// second projects @B, whose type at i64 is i1. Selection reads !T off the
// second alone, as the impl's verifier counts it (rustc's E0207 skips a
// projection of the trait being implemented): !T is i1.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @B[!S] {
  trait.assoc_type @X
}
trait.impl private @B_i64 for @B[i64] {
  trait.assoc_type @X = i1
}

trait.trait private @A[!S] {
  trait.assoc_type @Output
}
trait.impl private @A_gen for @A[!S] where [@B[!S], !trait.proj<@A[!S], "Output"> = !T, !trait.proj<@B[!S], "X"> = !T] {
  trait.assoc_type @Output = !T
}

func.func private @need(!trait.claim<@A[i64]>)

// CHECK-LABEL: func.func @main
// CHECK: trait.witness @[[PROOF:.*]] for @A[i64]
// CHECK: trait.proof private @[[PROOF]] proves @A_gen[!trait.poly<0> = i64, !trait.poly<1> = i1] for @A[i64]
func.func @main() {
  %c = trait.allege @A[i64]
  func.call @need(%c) : (!trait.claim<@A[i64]>) -> ()
  return
}
