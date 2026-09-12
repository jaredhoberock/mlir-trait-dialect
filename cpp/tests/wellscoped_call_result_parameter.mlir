// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A call whose result the arguments do not determine spells a type parameter in
// the caller's body. When the caller binds none -- it is no template, so nothing
// clones it and nothing supplies an argument -- that parameter has no source,
// and the call site is named at the entry, before the rounds try to rewrite it.
// (A rewritable call that survives the rounds for a reason the caller's scope
// admits, a generic callee with no body to clone, is named at the exit instead;
// invalid_call_to_an_external_generic_callee pins that refusal.)

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

func.func private @foo(!trait.poly<0>) -> !trait.poly<1>

// expected-error@+1 {{type parameter '!trait.poly<9>' is outside the signature scope of @main}}
func.func @main(%x: i64) {
  // expected-note@+1 {{mentioned here}}
  %r = trait.func.call @foo(%x) {type_params = [!trait.poly<0>, !trait.poly<1>], type_args = [i64, !trait.poly<9>]} : (i64) -> !trait.poly<9>
  return
}
