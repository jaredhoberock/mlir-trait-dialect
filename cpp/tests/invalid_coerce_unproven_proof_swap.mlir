// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The deep no-proof-swap clause runs unchanged under the marker. A marked
// coerce compares modulo a receipt but may not exchange one: a proof standing
// on the result that the input never carried is a swap, refused even though the
// pending unification would otherwise reconcile the endpoints modulo the
// receipt.

trait.trait @Bound[!trait.poly<0>] {
  trait.assoc_type @Item
}

func.func @receipt_exchange(%x: !trait.claim<@Bound[!trait.proj<@Bound[i64], "Item">]>)
    -> !trait.claim<@Bound[i64] by @p> {
  // expected-error @below {{may not swap the proof backing}}
  %y = trait.coerce %x
    : !trait.claim<@Bound[!trait.proj<@Bound[i64], "Item">]>
    to !trait.claim<@Bound[i64] by @p> unproven
  return %y : !trait.claim<@Bound[i64] by @p>
}
