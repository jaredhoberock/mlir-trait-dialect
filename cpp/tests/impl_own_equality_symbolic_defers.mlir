// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A polymorphic impl asserts its own equality Self::Output = Acc, where Acc is a
// free parameter it also binds Output to. Both endpoints stay symbolic, so the
// impl-verification check cannot decide the equality and defers it to instantiation, exactly
// as a symbolic trait-header equality requirement defers. The impl verifies clean.

!S = !trait.poly<0>
!Acc = !trait.poly<1>

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
}

// CHECK: trait.impl @FoldFn_gen for @FoldFn[!trait.poly<0>]where [!trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<1>]
trait.impl @FoldFn_gen for @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !Acc] {
  trait.assoc_type @Output = !Acc
}
