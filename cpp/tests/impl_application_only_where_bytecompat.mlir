// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --emit-bytecode | mlir-opt | FileCheck %s

// Generalizing the impl `where` clause to a mixed predicate list does not perturb
// an application-only impl: a clause of trait applications alone prints and
// parses byte-identically to the trait application array it generalizes -- one
// bracketed list, no equality section. Both the textual and the bytecode
// round-trips are unchanged.

!S = !trait.poly<0>

trait.trait @Eq[!S] {}

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
}

// CHECK: trait.impl @A_impl for @FoldFn[i32]where [@Eq[i32]]
trait.impl @A_impl for @FoldFn[i32] where [@Eq[i32]] {
  trait.assoc_type @Output = i32
}
