// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The method's declared result is a sibling projection, @Sib[Self]::Elem, and
// two impls bind @Sib[i64] -- so no unique module-visible impl answers it and a
// lookup over the module would decline. What answers it is the evidence the
// call holds: the receiver claim is proven by @p, whose subproof at the index
// of the trait's own requirement names @Sib_i64, and that impl's binding is
// what carries the declared result to the f32 the call spells.

!S = !trait.poly<0>

trait.trait private @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64 for @Sib[i64] {
  trait.assoc_type @Elem = f32
}

trait.impl private @Sib_i64_twin for @Sib[i64] {
  trait.assoc_type @Elem = i16
}

trait.trait private @Host[!S] where [@Sib[!S]] {
  func.func private @get(!S) -> !trait.proj<@Sib[!S], "Elem">
}

trait.impl private @Host_i64 for @Host[i64]
    witnesses [#trait<witness !trait.proj<@Sib[i64], "Elem"> = f32 by @Sib_i64>] {
  func.func @get(%x: i64) -> f32 {
    %r = ub.poison : f32
    return %r : f32
  }
}

trait.proof private @p proves @Host_i64 for @Host[i64] given [@Sib_i64]

// CHECK-LABEL: func.func @main
// CHECK: trait.method.call
func.func @main(%x: i64) -> f32 {
  %c = trait.witness @p for @Host[i64]
  %r = trait.method.call %c @Host[i64]::@get(%x)
    : (i64) -> f32
    by @p
  return %r : f32
}
