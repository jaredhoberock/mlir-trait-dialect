// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A derive supplies one claim per entry of its impl's where clause, each the
// entry at the stated arguments, so an equality claim supplied for an
// application entry is refused at its position.

!T0 = !trait.poly<0>
trait.trait private @Trait(%self: !trait.claim<@Trait[!T0]>) {}
trait.impl private @Trait_impl_i32(%self: !trait.claim<@Trait[i32]>) {}
trait.impl private @Trait_impl_tuple(%self: !trait.claim<@Trait[tuple<!T0>]>, %trait: !trait.claim<@Trait[!T0]>) {}

func.func @f(%e: !trait.claim<i32 = i32>) -> !trait.claim<@Trait[tuple<i32>]> {
  // expected-error @below {{premise 0 of impl '@Trait_impl_tuple' is '!trait.claim<@Trait[i32]>', and the derive supplies '!trait.claim<i32 = i32>'}}
  %d = trait.derive @Trait[tuple<i32>] from @Trait_impl_tuple[i32] given(%e) : (!trait.claim<i32 = i32>)
  return %d : !trait.claim<@Trait[tuple<i32>]>
}
