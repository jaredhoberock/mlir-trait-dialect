// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Tests that the TraitOp verifier accepts where-clause requirements that
// mention a GAT poly var (from an AssocTypeOp's type_params) rather than
// a trait type parameter.

!S = !trait.poly<0>
!T = !trait.poly<1>

// CHECK-LABEL: trait @Printable
trait.trait @Printable[!S] {}

// CHECK-LABEL: trait @Container
// CHECK: trait.assoc_type @Item<[!trait.poly<1>]>
trait.trait @Container[!S] where [@Printable[!T]] {
  trait.assoc_type @Item<[!T]>
}

// A where clause that bounds a non-GAT associated type via a projection:
// where Self::Item : Printable
// CHECK-LABEL: trait @Iterable
// CHECK: trait.assoc_type @Item
trait.trait @Iterable[!S] where [@Printable[!trait.proj<@Iterable[!S], "Item">]] {
  trait.assoc_type @Item
}
