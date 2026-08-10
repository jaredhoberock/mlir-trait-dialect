// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' | FileCheck %s

// The polymorph-erasing barrier maps a surviving claim-respell coerce to
// nothing: its claim-typed operand and its cited equality are proof material
// that carries no runtime value, so an unused respell erases outright and the
// function body is empty.

trait.trait @Bound[!trait.poly<0>] {
}

trait.trait @Assoc[!trait.poly<0>] {
  trait.assoc_type @Output
}
trait.impl @Assoc_impl for @Assoc[i64] {
  trait.assoc_type @Output = i64
}

// CHECK-LABEL: func.func @unused_claim_respell
// CHECK-NOT: trait.coerce
// CHECK-NEXT: return
func.func @unused_claim_respell(%b: !trait.claim<@Bound[i64]>,
                                %eq: !trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>) {
  %c = trait.coerce %b : !trait.claim<@Bound[i64]>
    to !trait.claim<@Bound[!trait.proj<@Assoc[i64], "Output">]>
    via (%eq) : (!trait.claim<!trait.proj<@Assoc[i64], "Output"> = i64>)
  return
}
