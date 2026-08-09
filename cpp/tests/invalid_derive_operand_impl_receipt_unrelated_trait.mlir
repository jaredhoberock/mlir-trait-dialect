// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' -verify-diagnostics

// The same forged receipt (an impl of an unrelated trait named as a `by @impl`
// proof) reaches a conditional impl's assumption: @CondImpl requires
// @Other[@HasPart[i64]::Part], and the drift ground-resolves that projection to
// f32, so the derive itself would go through. The receipt on %op still cannot
// specialize @HasPart_i64's self claim to @Other, so the audit at the receipt's
// declaration refuses it before any derivation can launder it into a proof.

trait.trait @HasPart[!trait.poly<0>] {
  trait.assoc_type @Part
}
trait.impl @HasPart_i64 for @HasPart[i64] {
  trait.assoc_type @Part = f32
}
trait.trait @Other[!trait.poly<0>] {}
trait.impl @Other_f32 for @Other[f32] {}
trait.trait @Tr[!trait.poly<0>] {}
trait.impl @CondImpl for @Tr[!trait.poly<1>]
    where [@Other[!trait.proj<@HasPart[i64], "Part">]] {}

// expected-error @below {{declared claim in signature has an invalid proof: trait mismatch: expected @HasPart, but found @Other}}
func.func private @template(
    %op: !trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">] by @HasPart_i64>
) -> !trait.claim<@Tr[!trait.poly<7>]> {
  %d = trait.derive @Tr[!trait.poly<7>] from @CondImpl given(%op)
    : (!trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">] by @HasPart_i64>)
  return %d : !trait.claim<@Tr[!trait.poly<7>]>
}
