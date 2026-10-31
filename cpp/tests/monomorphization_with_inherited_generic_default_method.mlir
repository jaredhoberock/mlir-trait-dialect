// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

trait.trait private @Trait(!trait.claim<@Trait[!trait.poly<0>]>) members [default @Trait.method]
func.func private @Trait.method(%self_claim: !trait.claim<@Trait[!trait.poly<0>]>, %self: !trait.poly<0>, %value: !trait.poly<1>) -> !trait.poly<1> {
  return %value : !trait.poly<1>
}

trait.impl private @Trait_impl_i64(%self: !trait.claim<@Trait[i64]>) members [@Trait.method -> @Trait.method] {
}

func.func @main() -> i32 {
  %self_claim0 = trait.witness @Trait_impl_i64 for @Trait[i64]
  %zero0 = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %x = trait.func.call @Trait.method(%self_claim0, %zero0, %one) : (!trait.claim<@Trait[i64] by @Trait_impl_i64>, i64, i64) -> i64

  %self_claim1 = trait.witness @Trait_impl_i64 for @Trait[i64]
  %zero1 = arith.constant 0 : i64
  %true = arith.constant true
  %y = trait.func.call @Trait.method(%self_claim1, %zero1, %true) : (!trait.claim<@Trait[i64] by @Trait_impl_i64>, i64, i1) -> i1

  %result = arith.constant 0 : i32
  return %result : i32
}

// CHECK-LABEL: func.func private @Trait.method_{{h[0-9a-f]+}}(
// CHECK-SAME: i64) -> i64
// CHECK-LABEL: func.func private @Trait.method_{{h[0-9a-f]+}}(
// CHECK-SAME: i1) -> i1
// CHECK-LABEL: func.func @main()
// CHECK: call @Trait.method_{{h[0-9a-f]+}}(
// CHECK-SAME: i64
// CHECK: call @Trait.method_{{h[0-9a-f]+}}(
// CHECK-SAME: i1
// CHECK-NOT: trait.func.call
