// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An impl's own equality assumption has a sibling-projection endpoint whose only
// impl is CONDITIONAL on a premise no impl proves (@Sib_i8 needs @X[i8]). No
// declared premise resolves that endpoint, so it stays symbolic at birth: a
// non-ground projection cannot be decided against the impl's bindings, and the
// birth check defers rather than refuse. Its correctness is established where the
// impl is selected and, for consumed evidence, at the use site. A blind lookup
// would have resolved the conditional impl anyway and over-refused f32 != i64;
// the birth check no longer enumerates candidates, so it defers.

!S = !trait.poly<0>

trait.trait @X[!S] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i8 for @Sib[i8] where [@X[i8]] {
  trait.assoc_type @Elem = i64
}

trait.trait @Host[!S] {}

// CHECK: trait.impl @Host_i8
trait.impl @Host_i8 for @Host[i8] where [!trait.proj<@Sib[i8], "Elem"> = f32] {
}
