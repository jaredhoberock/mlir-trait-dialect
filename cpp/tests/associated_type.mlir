// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: trait private @Iterator
// CHECK-SAME: assoc {Item = 0} members [@Iterator.next]
// CHECK: func.func private @Iterator.next(!trait.claim<@Iterator[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Iterator[!trait.poly<0>], "Item">

!S = !trait.poly<0>
trait.trait private @Iterator(!trait.claim<@Iterator[!S]>) assoc {Item = 0} members [@Iterator.next]
func.func private @Iterator.next(!trait.claim<@Iterator[!S]>, !S) -> !trait.proj<@Iterator[!S], "Item">

// CHECK-LABEL: trait.impl private @Iterator_impl(%self: !trait.claim<@Iterator[i32]>)
// CHECK-SAME: bindings {Item = i64} members [@Iterator.next -> @Iterator_impl.next]
// CHECK: func.func private @Iterator_impl.next

trait.impl private @Iterator_impl(%self_claim: !trait.claim<@Iterator[i32]>) bindings {Item = i64} members [@Iterator.next -> @Iterator_impl.next] {
}
func.func private @Iterator_impl.next(%self_claim: !trait.claim<@Iterator[i32]>, %self: i32) -> i64 {
  %c = arith.constant 42 : i64
  return %c : i64
}
