// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not=error

// An impl method cites a where-clause entry spelling a projection by
// position and passes it to a method whose instance spells the projection
// resolved. Extracted for the instance a call wants, the citation reads the
// subproof the self proof names at the entry's position, spelled with its
// projection resolved as the rest of the instance is stamped, so the call meets
// the instance its type arguments and evidence name with no mismatch reported
// along the way.

!S = !trait.poly<0>
!F = !trait.poly<1>
!X = !trait.poly<2>

trait.trait private @Fn(%self: !trait.claim<@Fn[!F, !X]>) {
  trait.method @call(!F, !X) -> i64
}

trait.impl private @Fn_i8(%self: !trait.claim<@Fn[i8, i64]>) {
  trait.method @call(%f: i8, %x: i64) -> i64 {
    trait.return %x : i64
  }
}

trait.trait private @Tr(%self: !trait.claim<@Tr[!S]>) {
  trait.assoc_type @Root
  trait.method @go(!S, !F, !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>) -> i64
}

trait.impl private @Tr_i32(%self: !trait.claim<@Tr[i32]>) {
  trait.assoc_type @Root = i64
  trait.method @go(%s: i32, %f: !F, %c: !trait.claim<@Fn[!F, i64]>) -> i64 {
    %seven = arith.constant 7 : i64
    trait.return %seven : i64
  }
}

trait.trait private @U(%self: !trait.claim<@U[!S, !F]>) {
  trait.method @u(!S, !F) -> i64
}

trait.impl private @U_gen(%self: !trait.claim<@U[!S, !F]>, %tr: !trait.claim<@Tr[!S]>, %fn: !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>) {
  trait.method @u(%s: !S, %f: !F) -> i64 {
    %r = trait.method.call %tr @Tr[!S]::@go(%s, %f, %fn)
      : (!S, !F, !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>) -> i64
    trait.return %r : i64
  }
}

func.func @main(%s: i32, %f: i8) -> i64 {
  %p = trait.allege @U[i32, i8]
  %r = trait.method.call %p @U[i32, i8]::@u(%s, %f) : (i32, i8) -> i64
  return %r : i64
}

// CHECK: func.func private @[[GO:Tr_i32_h[0-9a-f]+_go]](%{{.*}}: i32, %{{.*}}: i8) -> i64
// CHECK: func.func private @[[U:U_gen_[_a-z0-9]*]](%{{.*}}: i32, %{{.*}}: i8) -> i64
// CHECK-NEXT: call @[[GO]](
// CHECK: func.func @main
// CHECK: call @[[U]](
