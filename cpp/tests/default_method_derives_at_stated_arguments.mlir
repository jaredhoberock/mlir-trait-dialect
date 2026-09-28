// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A trait default method derives through an impl at stated arguments whose
// key is the impl's own parameter, spelled with the label the trait's own
// parameter also carries. Cloning the method into an impl substitutes the
// trait's parameter in the argument and leaves the key the impl's, and the
// instance lowers through the cited impl.

!S = !trait.poly<0>

trait.trait private @Tr[!S] {
  func.func private @get(!S) -> i64
}

trait.impl private @Tr_tuple for @Tr[tuple<!S>] where [@Tr[!S]] {
  func.func @get(%x: tuple<!S>) -> i64 {
    %c = arith.constant 35 : i64
    return %c : i64
  }
}

trait.trait private @Wrap[!S] where [@Tr[!S]] {
  func.func @wrapped(%x: tuple<!S>) -> i64 {
    %t = trait.assume 0 : !trait.claim<@Tr[!S]>
    %d = trait.derive @Tr[tuple<!S>] from @Tr_tuple[!S = !S] given(%t) : (!trait.claim<@Tr[!S]>)
    %r = trait.method.call %d @Tr[tuple<!S>]::@get(%x) : (tuple<!S>) -> i64
    return %r : i64
  }
}

trait.impl private @Tr_i32 for @Tr[i32] {
  func.func @get(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

trait.impl private @Wrap_i32 for @Wrap[i32] {}

func.func @main(%x: tuple<i32>) -> i64 {
  %w = trait.allege @Wrap[i32]
  %r = trait.method.call %w @Wrap[i32]::@wrapped(%x) : (tuple<i32>) -> i64
  return %r : i64
}

// CHECK: func.func private @[[GET:Tr_tuple_[_a-z0-9]*get[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK: func.func private @[[W:Wrap_i32_wrapped[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK-NEXT: call @[[GET]](
// CHECK: func.func @main
// CHECK: call @[[W]](
// CHECK-NOT: trait.
