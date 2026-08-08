// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The synthesized impl symbol name folds the where-clause equality entries into
// its hash. Two impls with the same self application and the same application
// assumptions but a distinguishing equality assumption synthesize distinct
// names, so both coexist; without the equality their names would collide (an
// identical unnamed pair is a duplicate-symbol error). An application-only impl
// keeps the byte-identical name it had before the clause could hold equalities:
// giving it that exact synthesized name elides the name on print.

!S = !trait.poly<0>

trait.trait @Eq[!S] {}

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
}

// CHECK: trait.impl for @FoldFn[!trait.poly<0>]where [@Eq[!trait.poly<0>], !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>]
trait.impl for @FoldFn[!S] where [@Eq[!S], !trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output = !S
}

// CHECK: trait.impl for @FoldFn[!trait.poly<0>]where [@Eq[!trait.poly<0>]] {
trait.impl for @FoldFn[!S] where [@Eq[!S]] {
  trait.assoc_type @Output = !S
}

// The name below is exactly what generateSymName synthesizes from this impl's
// self application and its single application assumption. That it is elided on
// print proves the equality arm never perturbs an application-only impl's name.
// CHECK: trait.impl for @FoldFn[i32]where [@Eq[i32]] {
trait.impl @FoldFn_impl_h5097411fec491d52 for @FoldFn[i32] where [@Eq[i32]] {
  trait.assoc_type @Output = i32
}
