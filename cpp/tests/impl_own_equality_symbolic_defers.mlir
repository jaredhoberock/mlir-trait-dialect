// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A polymorphic impl asserts its own equality Self::Output = Acc, where Acc is a
// header parameter it also binds Output to. Both endpoints stay symbolic, so the
// impl-verification check cannot decide the equality and defers it to instantiation, exactly
// as a symbolic trait-header equality requirement defers. The impl verifies clean.

!S = !trait.poly<0>
!Acc = !trait.poly<1>

trait.trait private @FoldFn[!S, !Acc] {
  trait.assoc_type @Output
}

// CHECK: trait.impl private @FoldFn_gen for @FoldFn[!trait.poly<0>, !trait.poly<1>] where [!trait.proj<@FoldFn[!trait.poly<0>, !trait.poly<1>], "Output"> = !trait.poly<1>]
trait.impl private @FoldFn_gen for @FoldFn[!S, !Acc] where [!trait.proj<@FoldFn[!S, !Acc], "Output"> = !Acc] {
  trait.assoc_type @Output = !Acc
}
