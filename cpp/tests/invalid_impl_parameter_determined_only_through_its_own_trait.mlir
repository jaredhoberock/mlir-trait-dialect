// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// !T stands in no header position; the one equality spelling it projects the
// impl's own trait application, @A[!S]::Output, which resolves through this
// impl's own binding of Output to !T. Reading the equality to determine !T
// would first need !T, so nothing constrains it (rustc's E0207, which skips a
// projection of the trait being implemented) and the impl is refused where it
// is declared.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @A[!S] {
  trait.assoc_type @Output
}

// expected-error @below {{type parameter '!trait.poly<1>' is not constrained by the impl's trait application or its where clause, so impl selection cannot determine it}}
trait.impl private @A_gen for @A[!S] where [@A[!S], !trait.proj<@A[!S], "Output"> = !T] {
  trait.assoc_type @Output = !T
}
