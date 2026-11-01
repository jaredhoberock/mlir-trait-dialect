// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A trait's member is a function leading with the trait's own application;
// one with a body is the default an impl may leave the member to, and an
// external one is required of every impl.

!S = !trait.poly<0>
// expected-error @below {{member 0, @T.m, is no function}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]

// -----

!S = !trait.poly<0>
// expected-error @below {{member @T.m does not lead with the trait's own application '!trait.claim<@T[!trait.poly<0>]>'}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!S) -> i64

// -----

// A trait lists each member once.

!S = !trait.poly<0>
// expected-error @below {{lists member @T.m twice}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m, @T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64

// -----

// The trait refuses its list before an impl standing ahead of it is judged
// against it.

!S = !trait.poly<0>
trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@T.m -> @I.m] {
}
func.func private @I.m(%self: !trait.claim<@T[i32]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
// expected-error @below {{lists member @T.m twice}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m, @T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64

// -----

// A trait's member is named under it, @<trait>.<name>.

!S = !trait.poly<0>
// expected-error @below {{member @Elsewhere is not named under the trait}}
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m, @Elsewhere]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
func.func private @Elsewhere(!trait.claim<@T[!S]>) -> i64

// -----

!S = !trait.poly<0>
// expected-error @below {{declares associated type 'A' with "two", which is no count of its own parameters}}
"trait.trait"() <{sym_name = "T", self = !trait.claim<@T[!S]>, requirements = [], members = [], assoc = {A = "two"}}> : () -> ()

// -----

!S = !trait.poly<0>
// expected-error @below {{states its own application as the unproven claim @T[...]}}
trait.trait private @T(!trait.claim<@U[!S]>)
