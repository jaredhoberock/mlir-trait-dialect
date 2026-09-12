// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The impl's associated type binding is spelled over the impl's own parameter,
// which carries the same label as the trait's self, and the method the trait
// declares returns the projection that binding answers. The impl's own
// obligations are read through what its self claim says its parameters take --
// poly<0> is poly<0> there -- and not through the trait-to-impl substitution,
// which would rewrite the binding into tuple<poly<0>> and leave the impl's copy
// of the method disagreeing with the trait's declaration.

trait.trait private @Tr[!trait.poly<0>] {
  trait.assoc_type @X
  func.func private @f(!trait.poly<0>) -> !trait.proj<@Tr[!trait.poly<0>], "X">
}
trait.impl private @I for @Tr[tuple<!trait.poly<0>>] {
  trait.assoc_type @X = !trait.poly<0>
  func.func @f(%s: tuple<!trait.poly<0>>) -> !trait.poly<0> {
    %e = builtin.unrealized_conversion_cast %s : tuple<!trait.poly<0>> to !trait.poly<0>
    return %e : !trait.poly<0>
  }
}

// CHECK: func.func private @[[CLONE:I_h[0-9a-f]+_f_h[0-9a-f]+]](%{{.*}}: tuple<i32>) -> i32
// CHECK: func.func @main(%{{.*}}: tuple<i32>) -> i32
// CHECK: call @[[CLONE]]
func.func @main(%s: tuple<i32>) -> i32 {
  %c = trait.allege @Tr[tuple<i32>]
  %r = trait.method.call %c @Tr[tuple<i32>]::@f(%s) : (tuple<i32>) -> !trait.proj<@Tr[tuple<i32>], "X">
  %o = trait.coerce %r : !trait.proj<@Tr[tuple<i32>], "X"> to i32 unproven
  return %o : i32
}
