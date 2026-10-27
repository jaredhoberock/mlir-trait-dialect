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

// A trait, an impl and a proof answer the same law as every other template:
// each may be written public, and erase refuses each one still public when it
// runs.
// expected-error @below {{'trait.trait' op is a public template}}
trait.trait @Show(%self: !trait.claim<@Show[!trait.poly<0>]>) {
}

// expected-error @below {{'trait.impl' op is a public template}}
trait.impl @Show_i32(%self: !trait.claim<@Show[i32]>) {
}

// expected-error @below {{'trait.proof' op is a public template}}
trait.proof @Show_i32_p {
  %d = trait.derive @Show[i32] from @Show_i32 given()
  trait.return %d : !trait.claim<@Show[i32]>
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

// A generic call left standing is work instantiation still owes, so the
// entry refuses it before the conversion erases anything under it, and the
// exit check never meets the callee it names.
func.func private @poly(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

func.func @main() -> i32 {
  %c = arith.constant 0 : i32
  // expected-error @+1 {{'trait.func.call' op is a generic call instantiate-monomorphs has not lowered before erasure}}
  %r = trait.func.call @poly(%c) : (i32) -> i32
  return %r : i32
}
