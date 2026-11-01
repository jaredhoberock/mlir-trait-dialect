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

!S = !trait.poly<0>
// expected-error @below {{declares associated type 'A' with "two", which is no count of its own parameters}}
"trait.trait"() <{sym_name = "T", self = !trait.claim<@T[!S]>, requirements = [], members = [], assoc = {A = "two"}}> : () -> ()

// -----

!S = !trait.poly<0>
// expected-error @below {{states its own application as the unproven claim @T[...]}}
trait.trait private @T(!trait.claim<@U[!S]>)
