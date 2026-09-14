// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A generic associated type's parameter is the key a projection's associated
// type argument is substituted for. A ground type standing in that list is no
// key: the substitution would rewrite every occurrence of that same type in the
// bound type, so a binding authored as tuple<i32, i32> would resolve to
// tuple<i64, i64> at a projection spelled <[i64]>. The declaration is refused
// where it stands, at the trait that declares the associated type and at the
// impl that binds it.

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

trait.trait private @Family[!trait.poly<0>] {
  // expected-error@+1 {{type parameter list holds 'i32', which is not a type variable}}
  trait.assoc_type @A<[i32]>
}

// -----

trait.trait private @Family[!trait.poly<0>] {
  trait.assoc_type @A<[!trait.poly<1>]>
}

trait.impl private @Family_i1 for @Family[i1] {
  // expected-error@+1 {{type parameter list holds 'i32', which is not a type variable}}
  trait.assoc_type @A<[i32]> = tuple<i32, i32>
}
