// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The synthesized impl symbol name folds the where-clause equality entries into
// its hash. Two impls with the same self application and the same application
// assumptions but a distinguishing equality assumption synthesize distinct
// names, so both coexist; without the equality their names would collide (an
// identical unnamed pair is a duplicate-symbol error).

!S = !trait.poly<0>

trait.trait private @Eq(%self: !trait.claim<@Eq[!S]>) {}

trait.trait private @FoldFn(%self: !trait.claim<@FoldFn[!S]>) {
  trait.assoc_type @Output
}

// CHECK: trait.impl private @FoldFn_impl(%self: !trait.claim<@FoldFn[!trait.poly<0>]>
trait.impl private @FoldFn_impl(%self: !trait.claim<@FoldFn[!S]>, %eq: !trait.claim<@Eq[!S]>, %output: !trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S>) {
  trait.assoc_type @Output = !S
}

// CHECK: trait.impl private @FoldFn_impl1(%self: !trait.claim<@FoldFn[!trait.poly<0>]>
trait.impl private @FoldFn_impl1(%self: !trait.claim<@FoldFn[!S]>, %eq: !trait.claim<@Eq[!S]>) {
  trait.assoc_type @Output = !S
}
