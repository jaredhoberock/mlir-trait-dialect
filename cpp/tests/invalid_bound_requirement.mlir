// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A binder quantifying nothing is a plain predicate spelled another way.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate binds at least one parameter}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [] -> @Marker[!trait.poly<0>]] {}

// -----

// A conclusion that spells no parameter the binder binds quantifies nothing.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{a bound predicate's conclusion spells none of the parameters it binds}}
// expected-error @below {{expected a predicate array}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.poly<1>] -> @Marker[!trait.poly<0>]] {}

// -----

// A binder may not rebind the trait's own parameter.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{bound requirement 0 binds '!trait.poly<0>', which is a parameter of trait '@Has'}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.poly<0>] -> @Marker[!trait.poly<0>]] {}

// -----

// A bound parameter stands only inside its binder.

trait.trait private @Marker[!trait.poly<0>] {}
// expected-error @below {{requirement 1 spells '!trait.poly<1>', which bound requirement 0 binds; a bound parameter stands only inside its binder}}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.poly<1>] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>], @Marker[!trait.poly<1>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
}

// -----

// Only a trait's requirement is quantified; an impl's premise binds nothing.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Has[!trait.poly<0>] {}
// expected-error @below {{where-clause entry 0 binds parameters of its own; only a trait's requirement is quantified}}
trait.impl private @Has_any for @Has[!trait.poly<0>] where [forall [!trait.poly<1>] -> @Marker[tuple<!trait.poly<0>, !trait.poly<1>>]] {}

// -----

// A bound requirement is not a claim a scope holds; it is selected at arguments.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Has[!trait.poly<0>] where [forall [!trait.poly<1>] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]] {
  trait.assoc_type @A<[!trait.poly<1>]>
  func.func @m(%x: !trait.poly<0>) -> !trait.poly<0> {
    // expected-error @below {{cites where-clause entry 0, which binds parameters of its own; select it with trait.project and its type arguments}}
    %b = trait.assume 0 : !trait.claim<@Marker[!trait.poly<0>]>
    return %x : !trait.poly<0>
  }
}
