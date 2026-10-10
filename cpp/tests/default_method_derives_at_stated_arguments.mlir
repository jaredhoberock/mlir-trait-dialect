// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait default method derives through an impl whose own parameter is
// spelled with the label the trait's own parameter also carries. Cutting the
// method for an impl substitutes the trait's parameter in the derived claim and
// leaves the cited impl's parameter its own, and the instance lowers through
// the cited impl.

!S = !trait.poly<0>

trait.trait private @Tr(%self: !trait.claim<@Tr[!S]>) {
  trait.method @get(!S) -> i64
}

trait.impl private @Tr_tuple(%self: !trait.claim<@Tr[tuple<!S>]>, %tr: !trait.claim<@Tr[!S]>) {
  trait.method @get(%x: tuple<!S>) -> i64 {
    %c = arith.constant 35 : i64
    trait.return %c : i64
  }
}

trait.trait private @Wrap(%self: !trait.claim<@Wrap[!S]>) -> !trait.claim<@Tr[!S]> {
  trait.method @wrapped(%x: tuple<!S>) -> i64 {
    %t = trait.project %self[0] : !trait.claim<@Wrap[!S]> -> !trait.claim<@Tr[!S]>
    %d = trait.derive @Tr[tuple<!S>] from @Tr_tuple[!trait.poly<0>] given(%t) : (!trait.claim<@Tr[!S]>)
    %r = trait.method.call %d @Tr[tuple<!S>]::@get(%x) : (tuple<!S>) -> i64
    trait.return %r : i64
  }
}

trait.impl private @Tr_i32(%self: !trait.claim<@Tr[i32]>) {
  trait.method @get(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}

trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) {
  %tr = trait.witness @Tr_i32 for @Tr[i32]
  trait.return %tr : !trait.claim<@Tr[i32] by @Tr_i32>
}

func.func @main(%x: tuple<i32>) -> i64 {
  %w = trait.allege @Wrap[i32]
  %r = trait.method.call %w @Wrap[i32]::@wrapped(%x) : (tuple<i32>) -> i64
  return %r : i64
}

// CHECK: func.func private @[[GET:Tr_tuple_[_a-z0-9]*get[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK: func.func private @[[W:Wrap_h[0-9a-f]+_wrapped]](%{{.*}}: tuple<i32>) -> i64
// CHECK-NEXT: call @[[GET]](
// CHECK: func.func @main
// CHECK: call @[[W]](
// CHECK-NOT: trait.
