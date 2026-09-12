// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics

// Erase's exit check reads the module with every template still standing, which
// is where a template nothing may collect is caught: a public one, and one a
// standing operation names from outside a template. Both are refused by name at
// the operation that carries them, never answered by deleting what mentions
// them.

// A public polymorphic function is a template collection may not take.
// expected-error @below {{'func.func' op is a public template}}
func.func @poly(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

// -----

// The visibility is judged wherever a template stands, so one inside a nested
// symbol table answers the same law as one at the top of the module.
module @inner {
  // expected-error @below {{'func.func' op is a public template}}
  func.func @poly(%x: !trait.poly<0>) -> !trait.poly<0> {
    return %x : !trait.poly<0>
  }
}

// -----

// A provenance attribute naming a template is a walker-visible reference that
// would keep the template alive through collection.
func.func private @poly(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

// expected-error @below {{'func.func' op names the template @poly from outside a template}}
func.func @main() -> i32 attributes {provenance = @poly} {
  %c = arith.constant 0 : i32
  return %c : i32
}

// -----

// A generic call left standing names its callee, so the callee cannot be
// collected and the call is what the check refuses. Its type arguments name the
// callee's parameters, so the same call is refused for carrying the trait type
// system outside a template as well.
func.func private @poly(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

func.func @main() -> i32 {
  %c = arith.constant 0 : i32
  // expected-error @+2 {{'trait.func.call' op still carries '!trait.poly<0>' after erasure}}
  // expected-error @+1 {{'trait.func.call' op names the template @poly from outside a template}}
  %r = trait.func.call @poly(%c) {type_params = [!trait.poly<0>], type_args = [i32]} : (i32) -> i32
  return %r : i32
}
