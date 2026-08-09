// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" -verify-diagnostics

// Control, not discriminator: the operand here is hand-spelled to the resolved
// form @Other[f32] and left unproven, so no respell round produced it. The
// reconciliation walk bridges only proven operands -- the respell that produces
// a drift only rewrites proven claims -- so it leaves this one untouched, and
// the derive verifier refuses it for the exact-spelling mismatch it is. The
// refusal stands both before and after the reconciliation walk exists; the walk
// rescues a writer-produced drift, never a hand-spelled mismatch.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T7 = !trait.poly<7>

trait.trait @HasPart[!S] {
  trait.assoc_type @Part
}

trait.impl @HasPart_i64 for @HasPart[i64] {
  trait.assoc_type @Part = f32
}

trait.trait @Other[!S] {}

trait.impl @Other_f32 for @Other[f32] {}

trait.trait @Tr[!S] {}

trait.impl @CondImpl for @Tr[!X] where [@Other[!trait.proj<@HasPart[i64], "Part">]] {}

func.func private @template(
  %op: !trait.claim<@Other[f32]>
) -> !trait.claim<@Tr[!T7]> {
  // expected-error @+1 {{assumption operand #0 has claim}}
  %d = trait.derive @Tr[!T7] from @CondImpl given(%op)
    : (!trait.claim<@Other[f32]>)
  return %d : !trait.claim<@Tr[!T7]>
}
