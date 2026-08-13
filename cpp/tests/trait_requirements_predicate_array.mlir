// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s --emit-bytecode | mlir-opt | FileCheck %s

// A trait's `where` clause carries a mix of arms: an application requirement
// and an equality requirement, in declaration order. An application-only array
// prints as a plain array of trait applications, so the application entry here
// prints unchanged; the equality entry rides beside it. The clause survives both
// the textual and the bytecode round-trip.

!S = !trait.poly<0>

trait.trait @Eq[!S] {}

// CHECK: trait.trait @FoldFn[!trait.poly<0>] where [@Eq[!trait.poly<0>], !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>]
trait.trait @FoldFn[!S] where [@Eq[!S], !trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output
}
