// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The marked (unproven) coerce stands in a pending judgment discharged at
// monomorphization; its reconciling equalities are not yet citable, so citing
// any equality is a contradiction the verifier refuses.

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

func.func @cites(%x: !trait.proj<@Fold[i64], "Item">,
                 %e: !trait.claim<!trait.proj<@Fold[i64], "Item"> = i64>) -> i64 {
  // expected-error @below {{may not cite equalities}}
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "Item"> to i64
    via (%e) : (!trait.claim<!trait.proj<@Fold[i64], "Item"> = i64>) unproven
  return %y : i64
}
