// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics

// A trait member returns a sibling trait's associated type. At the host impl's
// application that is the ground projection Sibling[i64]::Elem, which the
// impl's own bindings do not resolve, so the impl's member function keeps it.
// A member function writing its result as a rigid type is refused against
// that projection where the stage begins.

!S = !trait.poly<0>

trait.trait private @Sibling(!trait.claim<@Sibling[!S]>) assoc {Elem = 0}

trait.impl private @Sibling_i64(%self: !trait.claim<@Sibling[i64]>) bindings {Elem = i32} {
}

trait.trait private @Host(!trait.claim<@Host[!S]>) members [@Host.make]
func.func private @Host.make(!trait.claim<@Host[!S]>, !S) -> !trait.proj<@Sibling[!S], "Elem">

// expected-error @below {{member @Host.make has incompatible signature: expected '(i64) -> !trait.proj<@Sibling[i64], "Elem">' but found '(i64) -> i64'}}
trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) members [@Host.make -> @Host_i64.make] {
}
func.func private @Host_i64.make(%self: !trait.claim<@Host[i64]>, %x: i64) -> i64 {
  %r = ub.poison : i64
  return %r : i64
}

// -----

// The member function keeps the trait's projection and coerces the i64 its body computes
// at the boundary, citing @Sibling_i64 for Sibling[i64]::Elem = i64. The
// sibling binds Elem to i32, so the citation is refused.

!S = !trait.poly<0>

trait.trait private @Sibling(!trait.claim<@Sibling[!S]>) assoc {Elem = 0}

trait.impl private @Sibling_i64(%self: !trait.claim<@Sibling[i64]>) bindings {Elem = i32} {
}

trait.trait private @Host(!trait.claim<@Host[!S]>) members [@Host.make]
func.func private @Host.make(!trait.claim<@Host[!S]>, !S) -> !trait.proj<@Sibling[!S], "Elem">

trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) members [@Host.make -> @Host_i64.make] {
}
func.func private @Host_i64.make(%self: !trait.claim<@Host[i64]>, %x: i64) -> !trait.proj<@Sibling[i64], "Elem"> {
  %r = ub.poison : i64
  // expected-error @below {{impl '@Sibling_i64' binds the projection to 'i32', not the certified resolution 'i64'}}
  %elem = trait.witness proj_resolve !trait.proj<@Sibling[i64], "Elem"> resolves i64 by @Sibling_i64 : !trait.claim<!trait.proj<@Sibling[i64], "Elem"> = i64>
  %v = trait.coerce %r : i64 to !trait.proj<@Sibling[i64], "Elem"> via (%elem) : (!trait.claim<!trait.proj<@Sibling[i64], "Elem"> = i64>)
  return %v : !trait.proj<@Sibling[i64], "Elem">
}
