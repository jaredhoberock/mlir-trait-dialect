// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A trait's member is a function leading with the trait's own application;
// one the trait marks default has the body an impl may leave the member to,
// and every other is external, required of every impl.

!S = !trait.poly<0>
// expected-error @below {{member 0, @T.m, is no function}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]

// -----

!S = !trait.poly<0>
// expected-error @below {{member @T.m does not lead with the trait's own application '!trait.claim<@T[!trait.poly<0>]>'}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!S) -> i64

// -----

!S = !trait.poly<0>
// expected-error @below {{member @T.m is marked default and has no body}}
trait.trait private @T(!trait.claim<@T[!S]>) members [default @T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64

// -----

!S = !trait.poly<0>
// expected-error @below {{member @T.m has a body and is not marked default}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(%self: !trait.claim<@T[!S]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}

// -----

!S = !trait.poly<0>
// expected-error @below {{marks 0 member(s) as defaulted or not, and declares 1}}
"trait.trait"() <{sym_name = "T", self = !trait.claim<@T[!S]>, requirements = [], members = [@T.m], defaults = array<i1>, assoc = {}}> : () -> ()

// -----

!S = !trait.poly<0>
// expected-error @below {{declares associated type 'A' with "two", which is no count of its own parameters}}
"trait.trait"() <{sym_name = "T", self = !trait.claim<@T[!S]>, requirements = [], members = [], defaults = array<i1>, assoc = {A = "two"}}> : () -> ()

// -----

!S = !trait.poly<0>
// expected-error @below {{states its own application as the unproven claim @T[...]}}
trait.trait private @T(!trait.claim<@U[!S]>)
