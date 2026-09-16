// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T = !trait.poly<0>
trait.trait private @Get[!T] {
  // method returns the trait's type parameter
  func.func private @get() -> !T
}

trait.impl private for @Get[i32] {
  func.func @get() -> i32 {
    %c = arith.constant 0 : i32
    return %c : i32
  }
}

// The method's declared result is the trait's parameter, so a call writes its
// own result at the spelling that declaration gives it: @get through a claim
// for @Get[!A] returns !A, and nothing else. Monomorphization grounds !A := i32
// where the caller supplies it.
!A = !trait.poly<1>
func.func private @return_method_result(%claim: !trait.claim<@Get[!A]>) -> !A {
  %res = trait.method.call %claim @Get[!A]::@get()
    : () -> !A
  return %res : !A
}

func.func @bar() -> i32 {
  %a = trait.allege @Get[i32]
  %res = trait.func.call @return_method_result(%a)
    : (!trait.claim<@Get[i32]>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @bar() -> i32
// CHECK:       return {{.*}} : i32
// CHECK-NOT:   trait.trait
// CHECK-NOT:   trait.impl
// CHECK-NOT:   trait.func.call
// CHECK-NOT:   trait.method.call
// CHECK-NOT:   trait.allege
// CHECK-NOT:   trait.assume
// CHECK-NOT:   trait.witness
// CHECK-NOT:   trait.project
// CHECK-NOT:   !trait.poly<
