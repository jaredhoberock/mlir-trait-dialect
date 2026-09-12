// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A generic associated type has two lists of parameters standing over it, and a
// use supplies each from a different place: the header's parameters come from
// the application the impl is selected for, the associated type's own come from
// the projection's associated type arguments. A declaration that puts one label
// in both lists would have the projection's argument overwrite the
// application's, and a bound naming a label in neither has nothing to supply it,
// so the resolved type would carry a parameter no substitution reaches.

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

trait.trait private @Collide[!trait.poly<0>] {
  // expected-error@+1 {{type parameter '!trait.poly<0>' is already a parameter of trait '@Collide'}}
  trait.assoc_type @A<[!trait.poly<0>]>
}

// -----

trait.trait private @Family[!trait.poly<0>] {
  trait.assoc_type @A<[!trait.poly<1>]>
}

trait.impl private @Family_blanket for @Family[!trait.poly<0>] {
  // expected-error@+1 {{type parameter '!trait.poly<0>' is already a parameter of impl '@Family_blanket'}}
  trait.assoc_type @A<[!trait.poly<0>]> = !trait.poly<0>
}

// -----

trait.trait private @Container[!trait.poly<0>] {
  trait.assoc_type @Item<[!trait.poly<1>]>
}

trait.impl private @Container_i32 for @Container[i32] {
  // expected-error@+1 {{bound type mentions type parameter '!trait.poly<1>', which neither impl '@Container_i32' nor this associated type declares}}
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.poly<1>
}
