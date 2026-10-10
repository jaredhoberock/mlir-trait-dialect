// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An arith.select chooses between @Need[!P] derived over @One (7) and over
// @Two (9), each spelled through the projection !P, which resolves to i64. A
// select's result is a join of the two values it chooses between, and a claim
// names one proof, so a value whose proof would depend on the condition has no
// type: the result is refused where it stands, naming what each value carries,
// whichever way the flag goes, and neither value's proof is chosen.

!S = !trait.poly<0>
trait.trait private @Bound(%self: !trait.claim<@Bound[!S]>) { trait.method @value() -> i64 }
trait.impl private @One(%self: !trait.claim<@Bound[i64]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Two(%self: !trait.claim<@Bound[i64]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Need(%self: !trait.claim<@Need[!S]>) { trait.method @get() -> i64 }
trait.impl private @Need_from(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) {
  trait.method @get() -> i64 {
    %v = trait.method.call %b @Bound[!S]::@value() : () -> i64
    trait.return %v : i64
  }
}
!P = !trait.proj<@Trait[i64], "Output">
trait.trait private @Trait(%self: !trait.claim<@Trait[!S]>) { trait.assoc_type @Output }
trait.impl private @Trait_impl(%self: !trait.claim<@Trait[i64]>) { trait.assoc_type @Output = i64 }
func.func private @pick(%flag: i1) -> i64 {
  %b1 = trait.witness @One for @Bound[i64]
  %eq1 = trait.witness proj_resolve !P resolves i64 by @Trait_impl : !trait.claim<!P = i64>
  %c1 = trait.coerce %b1 : !trait.claim<@Bound[i64] by @One> to !trait.claim<@Bound[!P]> via (%eq1) : (!trait.claim<!P = i64>)
  %d1 = trait.derive @Need[!P] from @Need_from[!P] given(%c1) : (!trait.claim<@Bound[!P]>)
  %b2 = trait.witness @Two for @Bound[i64]
  %eq2 = trait.witness proj_resolve !P resolves i64 by @Trait_impl : !trait.claim<!P = i64>
  %c2 = trait.coerce %b2 : !trait.claim<@Bound[i64] by @Two> to !trait.claim<@Bound[!P]> via (%eq2) : (!trait.claim<!P = i64>)
  %d2 = trait.derive @Need[!P] from @Need_from[!P] given(%c2) : (!trait.claim<@Bound[!P]>)
  // expected-error @+2 {{unproven monomorphic claim '!trait.claim<@Need[!trait.proj<@Trait[i64], "Output">]>' after instantiate-monomorphs}}
  // expected-note-re @+1 {{control flow joins it from '!trait.claim<@Need[!trait.proj<@Trait[i64], "Output">] by @{{.*}}' and '!trait.claim<@Need[!trait.proj<@Trait[i64], "Output">] by @{{.*}}'}}
  %r = arith.select %flag, %d1, %d2 : !trait.claim<@Need[!P]>
  %v = trait.method.call %r @Need[!P]::@get() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %t = arith.constant true
  %v = func.call @pick(%t) : (i1) -> i64
  return %v : i64
}
