// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A binder quantifying nothing is a plain predicate spelled another way.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate binds at least one variable}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [] -> @Marker[!trait.poly<0>]] {}

// -----

// A conclusion that spells no variable the binder binds quantifies nothing.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate's conclusion spells none of the variables it binds}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[!trait.poly<0>]] {}

// -----

// A binder lists its variables by position; a declaration's parameter is not
// one.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate lists its variables in position order: expected !trait.bound<0>, found '!trait.poly<0>'}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.poly<0>] -> @Marker[!trait.poly<0>]] {}

// -----

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate lists its variables in position order: expected !trait.bound<0>, found '!trait.bound<1>'}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<1>] -> @Marker[tuple<!trait.poly<0>, !trait.bound<1>>]] {}

// -----

// A binder binds only the variables it lists.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate binds 1 variables, and #trait<application@Marker[tuple<!trait.bound<0>, !trait.bound<1>>]> spells '!trait.bound<1>'}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[tuple<!trait.bound<0>, !trait.bound<1>>]] {}

// -----

// A binder variable stands only inside its binder.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{spells a binder variable outside a bound predicate}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.bound<0>]>], @Marker[!trait.bound<0>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
}

// -----

// A declaration's parameter is never a binder variable, so no parameter of an
// impl can stand for the variable of the binder its evidence proves.

// expected-error @below {{type parameter '!trait.bound<0>' is a binder variable, which stands only inside a bound predicate}}
trait.trait private @Marker[!trait.bound<0>] {}

// -----

trait.trait private @Marker[!trait.poly<0>] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
}
// expected-error @below {{type parameter '!trait.bound<0>' is a binder variable, which stands only inside a bound predicate}}
trait.impl private @Has_tuple for @Has[tuple<!trait.bound<0>>]
    witnesses [#trait<witness requirement 0 by @Marker_i1>] {
  trait.assoc_type @A<[!trait.poly<1>]> = i1
}

// -----

// An associated type's own parameter is a declaration's parameter too.

!S = !trait.poly<0>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  // expected-error @below {{type parameter '!trait.bound<0>' is a binder variable, which stands only inside a bound predicate}}
  trait.assoc_type @A<[!trait.bound<0>]>
}

// -----

!S = !trait.poly<0>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by @Marker_i1>] {
  // expected-error @below {{type parameter '!trait.bound<0>' is a binder variable, which stands only inside a bound predicate}}
  trait.assoc_type @A<[!trait.bound<0>]> = i1
}

// -----

// A bound entry spells only the trait's parameters and its binder's variables:
// a premise over a parameter from nowhere would be satisfied by whatever
// same-labelled parameter a selecting scope holds.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{bound requirement 0 spells '!trait.poly<7>', which is neither a parameter of trait '@Has' nor a variable of its binder}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] where [@Marker[!trait.poly<7>]] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
}

// -----

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{bound requirement 0 spells '!trait.poly<7>', which is neither a parameter of trait '@Has' nor a variable of its binder}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[tuple<!trait.bound<0>, !trait.poly<7>>]] {}

// -----

// A bound conclusion is a requirement: it names its own trait only through a
// projection.

// expected-error @below {{bound requirement 0 concludes #trait<application@Has[!trait.bound<0>]>, which must not reference the current trait}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Has[!trait.bound<0>]] {}

// -----

// Only a trait's requirement is quantified; an impl's premise binds nothing.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Has[!trait.poly<0>] {}
// expected-error @below {{where-clause entry 0 binds variables of its own; only a trait's requirement is quantified}}
trait.impl private @Has_any for @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[tuple<!trait.poly<0>, !trait.bound<0>>]] {}

// -----

// A bound requirement is not a claim a scope holds; it is selected at arguments.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
  func.func @m(%x: !trait.poly<0>) -> !trait.poly<0> {
    // expected-error @below {{cites where-clause entry 0, which binds variables of its own; select it with trait.project and its type arguments}}
    %b = trait.assume 0 : !trait.claim<@Marker[!trait.poly<0>]>
    return %x : !trait.poly<0>
  }
}
