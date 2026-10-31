// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Both arms of an scf.if yield @Need[i64] derived over @Two, though @One
// proves @Bound[i64] too. A region's result repeats what its arms yield when
// they agree, so it carries their proof and the method called through it runs
// @Two's; selection, which would meet two impls, is never asked for it.

// CHECK: {{^}}9{{$}}

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
  %r = scf.if %flag -> !trait.claim<@Need[i64]> {
    %b = trait.witness @Two for @Bound[i64]
    %d = trait.derive @Need[i64] from @Need_from[i64] given(%b) : (!trait.claim<@Bound[i64] by @Two>)
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
