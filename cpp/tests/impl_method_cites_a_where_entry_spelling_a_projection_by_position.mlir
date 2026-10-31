// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not=error

// An impl's member function cites a where-clause entry naming a projection by
// position, its leading parameter there, and passes it to a member whose
// instance spells the projection resolved. A call of the member through a
// proven receiver passes the premise the receiver's proof derives the impl
// from at the entry's position, spelled with its projection resolved as the
// rest of the instance is stamped, so the call meets the instance its type
// arguments and evidence name with no mismatch reported along the way.

!S = !trait.poly<0>
!F = !trait.poly<1>
!X = !trait.poly<2>

trait.trait private @Fn(!trait.claim<@Fn[!trait.poly<0>, !trait.poly<1>]>) members [@Fn.call]
func.func private @Fn.call(!trait.claim<@Fn[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> i64

trait.impl private @Fn_i8(%self: !trait.claim<@Fn[i8, i64]>) members [@Fn.call -> @Fn_i8.call] {
}
func.func private @Fn_i8.call(%self: !trait.claim<@Fn[i8, i64]>, %f: i8, %x: i64) -> i64 {
  return %x : i64
}

trait.trait private @Tr(!trait.claim<@Tr[!S]>) assoc {Root = 0} members [@Tr.go]
func.func private @Tr.go(!trait.claim<@Tr[!S]>, !S, !F, !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>) -> i64

trait.impl private @Tr_i32(%self: !trait.claim<@Tr[i32]>) bindings {Root = i64} members [@Tr.go -> @Tr_i32.go] {
}
func.func private @Tr_i32.go(%self: !trait.claim<@Tr[i32]>, %s: i32, %f: !trait.poly<0>, %c: !trait.claim<@Fn[!trait.poly<0>, i64]>) -> i64 {
  %seven = arith.constant 7 : i64
  return %seven : i64
}

trait.trait private @U(!trait.claim<@U[!S, !F]>) members [@U.u]
func.func private @U.u(!trait.claim<@U[!S, !F]>, !S, !F) -> i64

trait.impl private @U_gen(%self: !trait.claim<@U[!S, !F]>, %tr: !trait.claim<@Tr[!S]>, %fn: !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>) members [@U.u -> @U_gen.u] {
}
func.func private @U_gen.u(%self: !trait.claim<@U[!S, !F]>, %tr: !trait.claim<@Tr[!S]>, %fn: !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>, %s: !S, %f: !F) -> i64 {
  %r = trait.func.call @Tr.go(%tr, %s, %f, %fn) : (!trait.claim<@Tr[!S]>, !S, !F, !trait.claim<@Fn[!F, !trait.proj<@Tr[!S], "Root">]>) -> i64
  return %r : i64
}

func.func @main(%s: i32, %f: i8) -> i64 {
  %p = trait.allege @U[i32, i8]
  %r = trait.func.call @U.u(%p, %s, %f) : (!trait.claim<@U[i32, i8]>, i32, i8) -> i64
  return %r : i64
}

// CHECK: func.func private @[[GO:Tr_i32.go_h[0-9a-f]+]](%{{.*}}: i32, %{{.*}}: i8) -> i64
// CHECK: func.func private @[[U:U_gen.u_h[0-9a-f]+]](%{{.*}}: i32, %{{.*}}: i8) -> i64
// CHECK-NEXT: call @[[GO]](
// CHECK: func.func @main
// CHECK: call @[[U]](
