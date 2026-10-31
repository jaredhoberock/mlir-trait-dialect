// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' | FileCheck %s

// A trait member returns a sibling trait's associated type whose only impl is
// conditional. At the host impl's application the member's result is the
// ground projection Sibling[i64]::Elem, which the impl's member function keeps;
// the body computes an i64 and coerces it at the boundary through the
// conditional impl's binding, a proj_resolve witness whose premise @Needs[i64]
// is the host impl's own where argument. The module verifies, and the member
// function agrees with the member where the stage begins.

!S = !trait.poly<0>

trait.trait private @Needs(!trait.claim<@Needs[!S]>)

trait.trait private @Sibling(!trait.claim<@Sibling[!S]>) assoc {Elem = 0}

trait.impl private @Sibling_i64(%self: !trait.claim<@Sibling[i64]>, %needs: !trait.claim<@Needs[i64]>) bindings {Elem = i64} {
}

trait.trait private @Host(!trait.claim<@Host[!S]>) assoc {Out = 0} members [@Host.make]
func.func private @Host.make(!trait.claim<@Host[!S]>, !S) -> !trait.proj<@Sibling[!S], "Elem">

// CHECK: trait.impl private @Host_i64
// CHECK: func.func private @Host_i64.make(%{{[^:]*}}: !trait.claim<@Host[i64]>, %{{[^:]*}}: !trait.claim<@Needs[i64]>, %{{[^:]*}}: i64) -> !trait.proj<@Sibling[i64], "Elem">
// CHECK: trait.coerce
trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>, %needs: !trait.claim<@Needs[i64]>) bindings {Out = i64} members [@Host.make -> @Host_i64.make] {
}
func.func private @Host_i64.make(%self: !trait.claim<@Host[i64]>, %needs: !trait.claim<@Needs[i64]>, %x: i64) -> !trait.proj<@Sibling[i64], "Elem"> {
  %r = ub.poison : i64
  %elem = trait.witness proj_resolve !trait.proj<@Sibling[i64], "Elem"> resolves i64 by @Sibling_i64 given(%needs) : (!trait.claim<@Needs[i64]>) : !trait.claim<!trait.proj<@Sibling[i64], "Elem"> = i64>
  %v = trait.coerce %r : i64 to !trait.proj<@Sibling[i64], "Elem"> via (%elem) : (!trait.claim<!trait.proj<@Sibling[i64], "Elem"> = i64>)
  return %v : !trait.proj<@Sibling[i64], "Elem">
}
