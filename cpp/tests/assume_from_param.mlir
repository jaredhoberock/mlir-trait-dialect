// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Test that trait.assume inside a standalone polymorphic function is replaced
// with the matching claim parameter during monomorphization.

!T = !trait.poly<0>

trait.trait @Foo[!T] {
  func.func private @foo(!T) -> !T
}

trait.impl @Foo_impl_i64 for @Foo[i64] {
  func.func @foo(%x: i64) -> i64 {
    return %x : i64
  }
}

func.func nested @standalone_poly(%c: !trait.claim<@Foo[!T]>, %x: !T) -> !T {
  %a = trait.assume @Foo[!T]
  %res = trait.method.call %a @Foo[!T]::@foo(%x)
    : (!T) -> !T
  return %res : !T
}

// CHECK-LABEL: func.func nested @standalone_poly
// CHECK-NOT: trait.assume
// CHECK: func.func @main
func.func @main() -> i64 {
  %c42 = arith.constant 42 : i64
  %claim = trait.allege @Foo[i64]
  %result = trait.func.call @standalone_poly(%claim, %c42)
    : (!trait.claim<@Foo[i64]>, i64) -> i64
  return %result : i64
}
