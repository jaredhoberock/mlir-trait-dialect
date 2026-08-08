// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// trait.derive matches each assumption operand against one of the cited impl's
// trait-application assumptions, so an assumption operand must be a
// trait-application claim. An equality claim carries no trait application to
// match, so it is refused at the operand as a verify diagnostic rather than
// asserting inside the arm-asserting accessor.

!T0 = !trait.poly<0>
trait.trait @Trait[!T0] {}
trait.impl @Trait_impl_i32 for @Trait[i32] {}
trait.impl @Trait_impl_tuple for @Trait[tuple<!T0>] where [@Trait[!T0]] {}

func.func @f(%e: !trait.claim<i32 = i32>) -> !trait.claim<@Trait[tuple<i32>]> {
  // expected-error @below {{operand #0 must be variadic of a trait-application '!trait.claim' type, but got '!trait.claim<i32 = i32>'}}
  %d = trait.derive @Trait[tuple<i32>] from @Trait_impl_tuple given(%e) : (!trait.claim<i32 = i32>)
  return %d : !trait.claim<@Trait[tuple<i32>]>
}
