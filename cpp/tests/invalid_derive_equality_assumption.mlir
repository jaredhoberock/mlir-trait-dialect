// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// trait.derive discharges the cited impl's application-arm assumptions, so each
// assumption operand must be a trait-application claim. An equality claim
// carries no trait application to match; the ODS operand type is loosened to any
// claim, so the arm is refused by the verifier with a located diagnostic naming
// the design law rather than asserting inside the arm-asserting accessor.

!T0 = !trait.poly<0>
trait.trait @Trait[!T0] {}
trait.impl @Trait_impl_i32 for @Trait[i32] {}
trait.impl @Trait_impl_tuple for @Trait[tuple<!T0>] where [@Trait[!T0]] {}

func.func @f(%e: !trait.claim<i32 = i32>) -> !trait.claim<@Trait[tuple<i32>]> {
  // expected-error @below {{assumption operand #0 ('!trait.claim<i32 = i32>') must be a trait-application claim; an equality claim is not a legal trait.derive operand}}
  %d = trait.derive @Trait[tuple<i32>] from @Trait_impl_tuple given(%e) : (!trait.claim<i32 = i32>)
  return %d : !trait.claim<@Trait[tuple<i32>]>
}
