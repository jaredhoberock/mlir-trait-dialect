// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Every symbol a witness names is checked: the impl it cites, a plain symbol
// reference the op's own symbol-use check reads, and the symbols standing in an
// equality endpoint of its result type, which the framework's type-symbol
// verification walks.

// The impl a projection-resolution witness cites must resolve to an impl.
module {
  trait.trait private @Eq(%self: !trait.claim<@Eq[!trait.poly<0>]>) {
    trait.assoc_type @Out
  }
  func.func @f() {
    // expected-error @+1 {{cannot find trait.impl '@NoSuchImpl' cited by the witness}}
    %w = trait.witness proj_resolve !trait.proj<@Eq[i32], "Out"> resolves i64 by @NoSuchImpl : !trait.claim<!trait.proj<@Eq[i32], "Out"> = i64>
    return
  }
}

// -----

// A dangling trait symbol in an endpoint of the result equality is refused,
// on a refl witness, which cites nothing else.
module {
  func.func @f() {
    // expected-error @+1 {{cannot find trait '@Undef'}}
    %w = trait.witness refl : !trait.claim<!trait.proj<@Undef[i32], "Out"> = !trait.proj<@Undef[i32], "Out">>
    return
  }
}
