// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e plain --entry-point-result=i64 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e projected --entry-point-result=i64 | FileCheck %s

// Both values an arith.select chooses between are @Need[i64] derived over
// @Two, though @One proves @Bound[i64] too, spelled plainly in @pick and
// through the projection !P, which resolves to i64, in @pick_projected. A
// select's result repeats what it chooses between when the two agree, so it
// carries their proof and the method called through it runs @Two's; selection,
// which would meet two impls, is never asked for it, and the select leaves
// with its claim at erasure.

// CHECK: {{^}}9{{$}}

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
  %b1 = trait.witness @Two for @Bound[i64]
  %d1 = trait.derive @Need[i64] from @Need_from[i64] given(%b1) : (!trait.claim<@Bound[i64] by @Two>)
  %b2 = trait.witness @Two for @Bound[i64]
  %d2 = trait.derive @Need[i64] from @Need_from[i64] given(%b2) : (!trait.claim<@Bound[i64] by @Two>)
  %r = arith.select %flag, %d1, %d2 : !trait.claim<@Need[i64]>
  %v = trait.method.call %r @Need[i64]::@get() : () -> i64
  return %v : i64
}
func.func private @pick_projected(%flag: i1) -> i64 {
  %b1 = trait.witness @Two for @Bound[i64]
  %eq1 = trait.witness proj_resolve !P resolves i64 by @Trait_impl : !trait.claim<!P = i64>
  %c1 = trait.coerce %b1 : !trait.claim<@Bound[i64] by @Two> to !trait.claim<@Bound[!P]> via (%eq1) : (!trait.claim<!P = i64>)
  %d1 = trait.derive @Need[!P] from @Need_from[!P] given(%c1) : (!trait.claim<@Bound[!P]>)
  %b2 = trait.witness @Two for @Bound[i64]
  %eq2 = trait.witness proj_resolve !P resolves i64 by @Trait_impl : !trait.claim<!P = i64>
  %c2 = trait.coerce %b2 : !trait.claim<@Bound[i64] by @Two> to !trait.claim<@Bound[!P]> via (%eq2) : (!trait.claim<!P = i64>)
  %d2 = trait.derive @Need[!P] from @Need_from[!P] given(%c2) : (!trait.claim<@Bound[!P]>)
  %r = arith.select %flag, %d1, %d2 : !trait.claim<@Need[!P]>
  %v = trait.method.call %r @Need[!P]::@get() : () -> i64
  return %v : i64
}
func.func @plain() -> i64 {
  %t = arith.constant true
  %v = func.call @pick(%t) : (i1) -> i64
  return %v : i64
}
func.func @projected() -> i64 {
  %t = arith.constant true
  %v = func.call @pick_projected(%t) : (i1) -> i64
  return %v : i64
}
