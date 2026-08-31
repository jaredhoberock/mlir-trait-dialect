// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -allow-unregistered-dialect -verify-diagnostics -split-input-file

// A witness implements SymbolUserAttrInterface, so a symbol-user walk reaching
// the attribute checks the symbols it names -- including those a syntactic walk
// cannot read. The witness rides a discardable attribute here so the automatic
// symbol-user verifier drives the interface.

// The impl a witness names must resolve to an impl.
module {
  trait.trait @Eq[!trait.poly<0>] {}
  // expected-error @+1 {{witness names '@NoSuchImpl', which does not resolve to an impl}}
  "test.holder"() {w = #trait<witness @Eq[i32] by @NoSuchImpl>} : () -> ()
}

// -----

// The trait symbol sealed inside an equality predicate's projection endpoint is
// walk-opaque -- TypeEqualityAttr's hand-written storage exposes no getAsKey, so
// a generic sub-element walk never reaches it. The witness's own symbol-use
// check reaches it through the claim accessor and refuses the dangling reference.
module {
  trait.trait @Whatever[!trait.poly<0>] {}
  trait.impl @Some for @Whatever[i32] {}
  // expected-error @+1 {{cannot find trait '@Undef'}}
  "test.holder"() {w = #trait<witness !trait.proj<@Undef[i32], "Out"> = i64 by @Some>} : () -> ()
}

// -----

// A `trait.witness` op producing an equality claim seals the same symbols in the
// endpoints of its RESULT type, where the framework's type-symbol verification
// skips them (the endpoints are walk-opaque, so the claim reports no symbol
// refs). The op's own `verifySymbolUses` verifies its result claim through the
// claim accessor, so a dangling endpoint symbol on a refl witness -- which cites
// nothing else -- is still refused rather than admitted.
module {
  func.func @f() {
    // expected-error @+1 {{cannot find trait '@Undef'}}
    %w = trait.witness refl : !trait.claim<!trait.proj<@Undef[i32], "Out"> = !trait.proj<@Undef[i32], "Out">>
    return
  }
}
