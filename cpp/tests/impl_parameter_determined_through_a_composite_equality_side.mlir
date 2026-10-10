// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

// @S_gen's header binds !T and leaves !U to its where clause, where !U stands
// inside a composite: @Marker[!T]::M = tuple<!U>. Once the header settles !T,
// the projection's inputs are known, and the type it names -- tuple<i1>,
// through @Marker_i64 -- is read against tuple<!U> as a header is read against
// a demand: !U is i1. So the impl verifies (rustc's E0207 accepts a parameter a
// projection with constrained inputs determines), and proving the allegation
// @S[i64] selects it at both arguments: the proof derives it given the equality
// premise at !U = i1.

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Marker(%self: !trait.claim<@Marker[!T]>) {
  trait.assoc_type @M
}
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) {
  trait.assoc_type @M = tuple<i1>
}

trait.trait private @S(%self: !trait.claim<@S[!T]>) {
  trait.assoc_type @Res
}
trait.impl private @S_gen(%self: !trait.claim<@S[!T]>, %marker: !trait.claim<@Marker[!T]>, %m: !trait.claim<!trait.proj<@Marker[!T], "M"> = tuple<!U>>) {
  trait.assoc_type @Res = !U
}

func.func private @need(!trait.claim<@S[i64]>)

// CHECK-LABEL: func.func @main
// CHECK: trait.witness @[[PROOF:.*]] for @S[i64]
// CHECK: trait.proof private @[[PROOF]] {
// CHECK: trait.derive @S[i64] from @S_gen[i64, i1] given({{.*}}) : (!trait.claim<@Marker[i64] by @Marker_i64>, !trait.claim<!trait.proj<@Marker[i64], "M"> = tuple<i1>>)
func.func @main() {
  %c = trait.allege @S[i64]
  func.call @need(%c) : (!trait.claim<@S[i64]>) -> ()
  return
}
