// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The impl's associated type binding is spelled over the impl's own parameter,
// which carries the same label as the trait's self, and the member the trait
// declares returns the projection that binding answers. The impl's own
// obligations are read through what its self claim says its parameters take --
// poly<0> is poly<0> there -- and not through the trait-to-impl substitution,
// which would rewrite the binding into tuple<poly<0>> and leave the impl's copy
// of the member disagreeing with the trait's declaration.

trait.trait private @Tr(!trait.claim<@Tr[!trait.poly<0>]>) assoc {X = 0} members [@Tr.f]
func.func private @Tr.f(!trait.claim<@Tr[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Tr[!trait.poly<0>], "X">
trait.impl private @I(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>) bindings {X = !trait.poly<0>} members [@Tr.f -> @I.f] {
}
func.func private @I.f(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>, %s: tuple<!trait.poly<0>>) -> !trait.poly<0> {
  %e = builtin.unrealized_conversion_cast %s : tuple<!trait.poly<0>> to !trait.poly<0>
  return %e : !trait.poly<0>
}

// CHECK: func.func private @[[CLONE:I.f_h[0-9a-f]+]](%{{.*}}: tuple<i32>) -> i32
// CHECK: func.func @main(%{{.*}}: tuple<i32>) -> i32
// CHECK: call @[[CLONE]]
func.func @main(%s: tuple<i32>) -> i32 {
  %c = trait.allege @Tr[tuple<i32>]
  %r = trait.func.call @Tr.f(%c, %s) : (!trait.claim<@Tr[tuple<i32>]>, tuple<i32>) -> !trait.proj<@Tr[tuple<i32>], "X">
  %e = trait.allege !trait.proj<@Tr[tuple<i32>], "X"> = i32
  %o = trait.coerce %r : !trait.proj<@Tr[tuple<i32>], "X"> to i32
    via (%e) : (!trait.claim<!trait.proj<@Tr[tuple<i32>], "X"> = i32>)
  return %o : i32
}
