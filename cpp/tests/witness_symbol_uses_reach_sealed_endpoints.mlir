// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -allow-unregistered-dialect -verify-diagnostics -split-input-file

// Every symbol a witness names is checked wherever the attribute rides: the
// impl it cites, which is a plain symbol reference the attribute's own
// symbol-user check reads, and the symbols standing in an equality endpoint,
// which are ordinary sub-elements the framework's type walk reaches. The
// witness rides a discardable attribute here so the automatic symbol-user
// verifier drives the interface.

// The impl a witness names must resolve to an impl.
module {
  trait.trait private @Eq[!trait.poly<0>] {}
  // expected-error @+1 {{witness names '@NoSuchImpl', which does not resolve to an impl}}
  "test.holder"() {w = #trait<witness @Eq[i32] by @NoSuchImpl>} : () -> ()
}

// -----

// A trait named inside an equality predicate's projection endpoint is an
// ordinary sub-element, so the framework's type walk reaches the projection
// there and its own symbol-use check refuses the dangling reference.
module {
  trait.trait private @Whatever[!trait.poly<0>] {}
  trait.impl private @Some for @Whatever[i32] {}
  // expected-error @+1 {{cannot find trait '@Undef'}}
  "test.holder"() {w = #trait<witness !trait.proj<@Undef[i32], "Out"> = i64 by @Some>} : () -> ()
}

// -----

// The same symbols stand in the endpoints of a `trait.witness` op's RESULT
// type, which the framework's type-symbol verification walks, so a dangling
// endpoint symbol on a refl witness -- which cites nothing else -- is refused
// there rather than admitted.
module {
  func.func @f() {
    // expected-error @+1 {{cannot find trait '@Undef'}}
    %w = trait.witness refl : !trait.claim<!trait.proj<@Undef[i32], "Out"> = !trait.proj<@Undef[i32], "Out">>
    return
  }
}
