// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Verifying a witness reads its resolved binding modulo the cited equality
// premises, and a premise whose one endpoint stands inside the other,
// !poly<0> = tuple<!poly<0>>, is one class of two members like any other: both
// sides of the comparison rewrite to the member the class stands for, which is
// the smaller of the two, so the comparison settles rather than expanding the
// spelling. The premise has no finite model, and nothing here reasons about
// that: an equality hypothesis is assumed wherever it stands, satisfiable or
// not, and it is the caller owing the premise who can never discharge it.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl private @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

func.func @f(%pre: !trait.claim<!S = tuple<!S>>) -> !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S> {
  %e = trait.witness proj_resolve !trait.proj<@Trait[!S], "Output"> resolves !S by @Trait_impl[!U = !S]
    given(%pre) : (!trait.claim<!S = tuple<!S>>)
    : !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S>
  return %e : !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S>
}

// -----

// The class the premise carves out does not make a disagreeing binding agree.
// @Trait_impl binds Output to the self type, the witness certifies i32, and
// neither endpoint of the premise reaches either spelling, so the binding check
// refuses exactly as it would with no premise at all.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl private @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

func.func @f(%pre: !trait.claim<!S = tuple<!S>>) -> !trait.claim<!trait.proj<@Trait[!S], "Output"> = i32> {
  // expected-error @below {{impl '@Trait_impl' binds the projection to '!trait.poly<0>', not the certified resolution 'i32'}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[!S], "Output"> resolves i32 by @Trait_impl[!U = !S]
    given(%pre) : (!trait.claim<!S = tuple<!S>>)
    : !trait.claim<!trait.proj<@Trait[!S], "Output"> = i32>
  return %e : !trait.claim<!trait.proj<@Trait[!S], "Output"> = i32>
}
