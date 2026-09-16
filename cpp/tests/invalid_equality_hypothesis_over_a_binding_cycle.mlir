// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Reading equalities as classes settles the equalities; it does not settle the
// associated type bindings underneath them. @Grow's Out is bound to a tuple over
// @Loop's Out and @Loop's Out to a tuple over @Grow's, so each resolution pass
// nests the spelling one level deeper whichever of the two the class rewrites
// toward. The hypothesis joining the two projections stands and contributes its
// one rewrite; the rewrite budget is what stops the bindings, and it reports the
// cycle as it did before any hypothesis was in scope.

!T = !trait.poly<0>
trait.trait private @Grow[!T] {
  trait.assoc_type @Out
}
trait.trait private @Loop[!T] {
  trait.assoc_type @Out
}

!U = !trait.poly<1>
trait.impl private @Grow_any for @Grow[!U] {
  trait.assoc_type @Out = tuple<!trait.proj<@Loop[!U], "Out">>
}
trait.impl private @Loop_any for @Loop[!U] {
  trait.assoc_type @Out = tuple<!trait.proj<@Grow[!U], "Out">>
}

!X = !trait.poly<2>
func.func private @callee(!trait.claim<@Grow[!X]>, !trait.claim<@Loop[!X]>)
    -> !trait.proj<@Grow[!X], "Out">

!Y = !trait.poly<3>
func.func @caller(%eq: !trait.claim<!trait.proj<@Grow[!Y], "Out"> = !trait.proj<@Loop[!Y], "Out">>)
    -> !trait.proj<@Loop[!Y], "Out"> {
  %g = trait.derive @Grow[!Y] from @Grow_any given()
  %l = trait.derive @Loop[!Y] from @Loop_any given()
  // expected-error @below {{projection normalization did not converge; check for cyclic associated type bindings}}
  %r = trait.func.call @callee(%g, %l)
    : (!trait.claim<@Grow[!Y]>, !trait.claim<@Loop[!Y]>) -> !trait.proj<@Loop[!Y], "Out">
  return %r : !trait.proj<@Loop[!Y], "Out">
}
