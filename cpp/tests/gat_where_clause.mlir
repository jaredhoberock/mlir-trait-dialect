// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Tests that the TraitOp verifier accepts where-clause requirements that
// mention a generic associated type's own parameter -- a label after the
// trait's, which the associated type's count declares -- rather than a trait
// type parameter.

!S = !trait.poly<0>
!T = !trait.poly<1>

// CHECK-LABEL: trait private @Printable
trait.trait private @Printable(!trait.claim<@Printable[!S]>)

// CHECK-LABEL: trait private @Container
// CHECK-SAME: -> !trait.claim<@Printable[!trait.poly<1>]> assoc {Item = 1}
trait.trait private @Container(!trait.claim<@Container[!S]>) -> !trait.claim<@Printable[!T]> assoc {Item = 1}

// A where clause that bounds a non-GAT associated type via a projection:
// where Self::Item : Printable
// CHECK-LABEL: trait private @Iterable
// CHECK-SAME: assoc {Item = 0}
trait.trait private @Iterable(!trait.claim<@Iterable[!S]>) -> !trait.claim<@Printable[!trait.proj<@Iterable[!S], "Item">]> assoc {Item = 0}
