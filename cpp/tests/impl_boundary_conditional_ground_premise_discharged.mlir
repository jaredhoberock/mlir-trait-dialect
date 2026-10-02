// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait method returns a sibling trait's associated type whose only impl is
// conditional. At the host impl's application the method's result is the
// ground projection Sibling[i64]::Elem, which the impl's signature keeps; the
// body computes an i64 and coerces it at the boundary through the conditional
// impl's binding, a proj_resolve witness whose premise @Needs[i64] is the host
// impl's own where argument. The module verifies.

!S = !trait.poly<0>

trait.trait private @Needs(%self: !trait.claim<@Needs[!S]>) {}

trait.trait private @Sibling(%self: !trait.claim<@Sibling[!S]>) {
  trait.assoc_type @Elem
}

trait.impl private @Sibling_i64(%self: !trait.claim<@Sibling[i64]>, %needs: !trait.claim<@Needs[i64]>) {
  trait.assoc_type @Elem = i64
}

trait.trait private @Host(%self: !trait.claim<@Host[!S]>) {
  trait.assoc_type @Out
  trait.method @make(!S) -> !trait.proj<@Sibling[!S], "Elem">
}

// CHECK: trait.impl private @Host_i64
// CHECK: trait.method @make(%{{.*}}: i64) -> !trait.proj<@Sibling[i64], "Elem">
// CHECK: trait.coerce
trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>, %needs: !trait.claim<@Needs[i64]>) {
  trait.assoc_type @Out = i64
  trait.method @make(%x: i64) -> !trait.proj<@Sibling[i64], "Elem"> {
    %r = ub.poison : i64
    %elem = trait.witness proj_resolve !trait.proj<@Sibling[i64], "Elem"> resolves i64 by @Sibling_i64 given(%needs) : (!trait.claim<@Needs[i64]>) : !trait.claim<!trait.proj<@Sibling[i64], "Elem"> = i64>
    %v = trait.coerce %r : i64 to !trait.proj<@Sibling[i64], "Elem"> via (%elem) : (!trait.claim<!trait.proj<@Sibling[i64], "Elem"> = i64>)
    trait.return %v : !trait.proj<@Sibling[i64], "Elem">
  }
}
