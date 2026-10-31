// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// The arms of an scf.if yield @Need[i64] derived over @One (7) and over @Two
// (9). A claim names one proof, so a value whose proof would depend on the arm
// taken has no type: the result is refused where it stands, naming what each
// arm yields, whichever way the flag goes, and no arm's proof is chosen.

!S = !trait.poly<0>
trait.trait private @Bound(!trait.claim<@Bound[!S]>) members [@Bound.value]
func.func private @Bound.value(!trait.claim<@Bound[!S]>) -> i64
trait.impl private @One(%self: !trait.claim<@Bound[i64]>) members [@Bound.value -> @One.value] {
}
func.func private @One.value(%self: !trait.claim<@Bound[i64]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Two(%self: !trait.claim<@Bound[i64]>) members [@Bound.value -> @Two.value] {
}
func.func private @Two.value(%self: !trait.claim<@Bound[i64]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.trait private @Need(!trait.claim<@Need[!S]>) members [@Need.get]
func.func private @Need.get(!trait.claim<@Need[!S]>) -> i64
trait.impl private @Need_from(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) members [@Need.get -> @Need_from.get] {
}
func.func private @Need_from.get(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) -> i64 {
  %v = trait.func.call @Bound.value(%b) : (!trait.claim<@Bound[!S]>) -> i64
  return %v : i64
}
func.func private @pick(%flag: i1) -> i64 {
  // expected-error @+2 {{unproven monomorphic claim '!trait.claim<@Need[i64]>' after instantiate-monomorphs}}
  // expected-note-re @+1 {{control flow joins it from '!trait.claim<@Need[i64] by @{{.*}}' and '!trait.claim<@Need[i64] by @{{.*}}'}}
  %r = scf.if %flag -> !trait.claim<@Need[i64]> {
    %b = trait.witness @One for @Bound[i64]
    %d = trait.derive @Need[i64] from @Need_from[i64] given(%b) : (!trait.claim<@Bound[i64] by @One>)
    scf.yield %d : !trait.claim<@Need[i64]>
  } else {
    %b = trait.witness @Two for @Bound[i64]
    %d = trait.derive @Need[i64] from @Need_from[i64] given(%b) : (!trait.claim<@Bound[i64] by @Two>)
    scf.yield %d : !trait.claim<@Need[i64]>
  }
  %v = trait.func.call @Need.get(%r) : (!trait.claim<@Need[i64]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %f = arith.constant false
  %v = func.call @pick(%f) : (i1) -> i64
  return %v : i64
}
