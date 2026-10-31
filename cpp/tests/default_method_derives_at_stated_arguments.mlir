// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait's default derives through an impl whose own parameter is spelled
// with the label the trait's own parameter also carries. Cutting the default
// for a receiver substitutes the trait's parameter in the derived claim and
// leaves the cited impl's parameter its own, and the instance lowers through
// the cited impl.

!S = !trait.poly<0>

trait.trait private @Tr(!trait.claim<@Tr[!S]>) members [@Tr.get]
func.func private @Tr.get(!trait.claim<@Tr[!S]>, !S) -> i64

trait.impl private @Tr_tuple(%self: !trait.claim<@Tr[tuple<!S>]>, %tr: !trait.claim<@Tr[!S]>) members [@Tr.get -> @Tr_tuple.get] {
}
func.func private @Tr_tuple.get(%self: !trait.claim<@Tr[tuple<!S>]>, %tr: !trait.claim<@Tr[!S]>, %x: tuple<!S>) -> i64 {
  %c = arith.constant 35 : i64
  return %c : i64
}

trait.trait private @Wrap(!trait.claim<@Wrap[!S]>) -> !trait.claim<@Tr[!S]> members [default @Wrap.wrapped]
func.func private @Wrap.wrapped(%self: !trait.claim<@Wrap[!S]>, %x: tuple<!S>) -> i64 {
  %t = trait.project %self[0] : !trait.claim<@Wrap[!S]> -> !trait.claim<@Tr[!S]>
  %d = trait.derive @Tr[tuple<!S>] from @Tr_tuple[!trait.poly<0>] given(%t) : (!trait.claim<@Tr[!S]>)
  %r = trait.func.call @Tr.get(%d, %x) : (!trait.claim<@Tr[tuple<!S>]>, tuple<!S>) -> i64
  return %r : i64
}

trait.impl private @Tr_i32(%self: !trait.claim<@Tr[i32]>) members [@Tr.get -> @Tr_i32.get] {
}
func.func private @Tr_i32.get(%self: !trait.claim<@Tr[i32]>, %x: i32) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) members [@Wrap.wrapped -> @Wrap.wrapped] {
  %tr = trait.witness @Tr_i32 for @Tr[i32]
  trait.return %tr : !trait.claim<@Tr[i32] by @Tr_i32>
}

func.func @main(%x: tuple<i32>) -> i64 {
  %w = trait.allege @Wrap[i32]
  %r = trait.func.call @Wrap.wrapped(%w, %x) : (!trait.claim<@Wrap[i32]>, tuple<i32>) -> i64
  return %r : i64
}

// CHECK: func.func private @[[GET:Tr_tuple.get_h[0-9a-f]+]](%{{.*}}: tuple<i32>) -> i64
// CHECK: func.func private @[[W:Wrap.wrapped_h[0-9a-f]+]](%{{.*}}: tuple<i32>) -> i64
// CHECK-NEXT: call @[[GET]](
// CHECK: func.func @main
// CHECK: call @[[W]](
// CHECK-NOT: trait.
