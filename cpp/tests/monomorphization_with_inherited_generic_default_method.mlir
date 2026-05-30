// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

trait.trait @Trait[!trait.poly<0>] {
  func.func nested @method(%self: !trait.poly<0>, %value: !trait.poly<1>) -> !trait.poly<1> {
    return %value : !trait.poly<1>
  }
}

trait.impl @Trait_impl_i64 for @Trait[i64] {
}

func.func @main() -> i32 {
  %self_claim0 = trait.witness @Trait_impl_i64 for @Trait[i64]
  %zero0 = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %x = trait.method.call %self_claim0 @Trait[i64]::@method(%zero0, %one)
    : (i64, i64) -> i64
    by @Trait_impl_i64

  %self_claim1 = trait.witness @Trait_impl_i64 for @Trait[i64]
  %zero1 = arith.constant 0 : i64
  %true = arith.constant true
  %y = trait.method.call %self_claim1 @Trait[i64]::@method(%zero1, %true)
    : (i64, i1) -> i1
    by @Trait_impl_i64

  %result = arith.constant 0 : i32
  return %result : i32
}

// CHECK-LABEL: func.func private @Trait_impl_i64_method{{.*}}(
// CHECK-SAME: i64) -> i64
// CHECK-LABEL: func.func private @Trait_impl_i64_method{{.*}}(
// CHECK-SAME: i1) -> i1
// CHECK-LABEL: func.func @main()
// CHECK: call @Trait_impl_i64_method{{.*}}(
// CHECK-SAME: i64
// CHECK: call @Trait_impl_i64_method{{.*}}(
// CHECK-SAME: i1
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.func.call
