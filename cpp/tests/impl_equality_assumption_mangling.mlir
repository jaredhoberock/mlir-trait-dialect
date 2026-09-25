// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The synthesized impl symbol name folds the where-clause equality entries into
// its hash. Two impls with the same self application and the same application
// assumptions but a distinguishing equality assumption synthesize distinct
// names, so both coexist; without the equality their names would collide (an
// identical unnamed pair is a duplicate-symbol error).

!S = !trait.poly<0>

trait.trait private @Eq[!S] {}

trait.trait private @FoldFn[!S] {
  trait.assoc_type @Output
}

// CHECK: trait.impl private for @FoldFn[!trait.poly<0>]where [@Eq[!trait.poly<0>], !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>]
trait.impl private for @FoldFn[!S] where [@Eq[!S], !trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output = !S
}

// CHECK: trait.impl private for @FoldFn[!trait.poly<0>]where [@Eq[!trait.poly<0>]] {
trait.impl private for @FoldFn[!S] where [@Eq[!S]] {
  trait.assoc_type @Output = !S
}
