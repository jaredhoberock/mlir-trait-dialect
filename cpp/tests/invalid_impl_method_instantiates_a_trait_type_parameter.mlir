// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics

// The trait declares @T.f over two type parameters of its own, so a caller
// supplies an argument for each. The impl's copy spells tuple<P> where the
// trait spells its parameter. The trait's own parameters become the copy's by
// position, so the trait's signature carried to the impl spells the bare
// parameter there, and the copy's tuple is another signature: the copy stands
// only at the tuples, and a call at i64 would reach no instance of it. An
// impl's copy renames the trait's type parameters, it does not instantiate
// them.

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.f]
func.func private @T.f(!trait.claim<@T[!trait.poly<0>]>, !trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>

// expected-error @below {{member @T.f has incompatible signature: expected '(i32, !trait.poly<0>, !trait.poly<1>) -> !trait.poly<0>' but found '(i32, tuple<!trait.poly<0>>, !trait.poly<1>) -> tuple<!trait.poly<0>>'}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@T.f -> @I.f] {
}
func.func private @I.f(%self: !trait.claim<@T[i32]>, %x: i32, %m: tuple<!trait.poly<0>>, %n: !trait.poly<1>) -> tuple<!trait.poly<0>> {
  return %m : tuple<!trait.poly<0>>
}
