// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A projection resolves when its own spelling picks the impl that serves it. A
// projection over a bare variable picks none: `@Ten` has exactly one impl here,
// but reaching it means narrowing `!W` to `i64`, which is inference's choice to
// make and not this spelling's meaning. So the derive's element demand
// `@Other[!W]::X` and the blanket's `@Ten[!W]::Element` stay two projections
// over an unknown base and the derive is refused -- where a lookup willing to
// narrow the projection would have resolved `@Ten[!W]::Element` to
// `@Other[i64]::X`, matched it against the demand, and bound `!W` to `i64` on
// the way.

!A = !trait.poly<0>
trait.trait private @Other[!A] {
  trait.assoc_type @X
}

!P = !trait.poly<1>
trait.trait private @Ten[!P] {
  trait.assoc_type @Element
}

trait.impl private @Ten_i64 for @Ten[i64] {
  trait.assoc_type @Element = !trait.proj<@Other[i64], "X">
}

!C = !trait.poly<2>
!E = !trait.poly<3>
trait.trait private @Get[!C, !E] {
}

!T = !trait.poly<4>
trait.impl private @Get_blanket for @Get[!T, !trait.proj<@Ten[!T], "Element">]
    where [@Ten[!T]] {
}

!W = !trait.poly<5>
func.func @not_determined(%ten: !trait.claim<@Ten[!W]>) {
  // expected-error @below {{type mismatch: expected '!trait.claim<@Get[!trait.poly<5>, !trait.proj<@Ten[!trait.poly<5>], "Element">]>' but found '!trait.claim<@Get[!trait.poly<5>, !trait.proj<@Other[!trait.poly<5>], "X">]>'}}
  %get = trait.derive @Get[!W, !trait.proj<@Other[!W], "X">]
    from @Get_blanket given(%ten) : (!trait.claim<@Ten[!W]>)
  return
}
