// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics

// A trait's declaration of a member says how many type parameters of its own a
// caller supplies. An impl's member function that binds fewer is a different
// declaration: a call naming an argument for the trait's parameter would have
// nothing in the function to name. The correspondence between the two is
// positional, so equal counts are what the comparison where the stage begins
// requires.

!S = !trait.poly<0>
!M = !trait.poly<1>

trait.trait private @Keep(!trait.claim<@Keep[!S]>) members [@Keep.keep]
func.func private @Keep.keep(!trait.claim<@Keep[!S]>, !S, !M) -> !S

// expected-error @below {{member @Keep.keep is implemented binding 0 type parameter(s) of its own, but trait '"Keep"' declares it with 1}}
trait.impl private @Keep_i64(%self: !trait.claim<@Keep[i64]>) members [@Keep.keep -> @Keep_i64.keep] {
}
func.func private @Keep_i64.keep(%self: !trait.claim<@Keep[i64]>, %x: i64, %m: i32) -> i64 {
  return %x : i64
}
