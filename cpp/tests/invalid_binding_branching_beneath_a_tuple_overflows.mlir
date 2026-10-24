// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds Item<U> to a pair of Item projections at two larger arguments, so
// each layer of the binding holds twice the projections of the one above it,
// whichever branch comes first. What a step answers is read one step deeper,
// so the first branch read reaches the depth limit, and the reading overflows
// there, named once by what it spells as written, before it reads the rest.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }

// -----

// One branch, doubling its argument: the chain reaches the depth limit and
// overflows there. Its partial normal form spells a tree of 2^128 leaves in a
// few hundred distinct types, so it is named by what it spells as written,
// never by the partial form.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }

// -----

// A binding that branches to a finite depth: @C[tuple<T>]'s Out is a pair of
// @C[T]'s Out at two larger arguments, and @C[i1]'s is its argument, so the
// demand thirty deep has a normal form of 2^30 leaves, nearly all of them
// distinct types, and reading it takes a step at each of the 2^31 - 1
// projections the branching spells. No chain reaches the depth limit; the
// reading would take more steps than the step budget, and it overflows there.

!T = !trait.poly<0>
trait.trait private @C(%s: !trait.claim<@C[!T]>) { trait.assoc_type @Out<[!trait.poly<1>]> }
trait.impl private @Z(%s: !trait.claim<@C[i1]>) { trait.assoc_type @Out<[!trait.poly<0>]> = !trait.poly<0> }
trait.impl private @S(%s: !trait.claim<@C[tuple<!T>]>) { trait.assoc_type @Out<[!trait.poly<1>]> = tuple<!trait.proj<@C[!T], "Out", [tuple<!trait.poly<1>>]>, !trait.proj<@C[!T], "Out", [tuple<!trait.poly<1>, !trait.poly<1>>]>> }
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@C[tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<i1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>], "Out", [i64]> = i64>': resolving its projections takes more than 262144 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@C[tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<i1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>], "Out", [i64]> = i64>) { return }
