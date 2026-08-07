// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// An equality-arm !trait.claim<!A = !B> prints and parses through the infix `=`
// delimiter. Both a concrete-endpoint equality and a projection-endpoint
// equality survive a print/parse round-trip; the endpoints keep their exact
// spellings and orientation.

!S = !trait.poly<0>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
  func.func private @get(!S) -> !trait.proj<@Trait[!S], "Output">
}

// CHECK-LABEL: func.func private @concrete_equality
// CHECK-SAME: !trait.claim<i32 = i64>
func.func private @concrete_equality(!trait.claim<i32 = i64>)

// CHECK-LABEL: func.func private @projection_equality
// CHECK-SAME: !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
func.func private @projection_equality(!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
