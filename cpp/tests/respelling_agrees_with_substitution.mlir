// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=IR

// A claim nested inside another claim receives its proof throughout
// specialization. The generic caller's call of @Hold's method returns a claim,
// so at the instance it computes evidence, which selection proves in place of a
// call; the caller lowers with no claim types or unrealized conversion casts
// left behind.

!T = !trait.poly<0>

trait.trait private @Ground(%self: !trait.claim<@Ground[!T]>) {}

trait.impl private @Ground_all(%self: !trait.claim<@Ground[!T]>) {}

trait.trait private @Hold(%self: !trait.claim<@Hold[!T]>) {
  trait.method @held() -> !T
}

trait.impl private @Hold_claim(%self: !trait.claim<@Hold[!trait.claim<@Ground[i32]>]>, %ground: !trait.claim<@Ground[i32]>) {
  trait.method @held() -> !trait.claim<@Ground[i32]> {
    trait.return %ground : !trait.claim<@Ground[i32]>
  }
}

func.func private @take(%h: !trait.claim<@Hold[!T]>) -> !T {
  %v = trait.method.call %h @Hold[!T]::@held() : () -> !T
  return %v : !T
}

func.func @test() {
  %h = trait.allege @Hold[!trait.claim<@Ground[i32]>]
  trait.func.call @take(%h)
    : (!trait.claim<@Hold[!trait.claim<@Ground[i32]>]>) -> !trait.claim<@Ground[i32]>
  return
}

// IR-NOT: trait.claim
// IR-NOT: builtin.unrealized_conversion_cast

// CHECK-NOT: @Hold_claim
// CHECK-LABEL: func.func private @take_
// CHECK-NEXT: return
// CHECK-LABEL: func.func @test()
// CHECK: call @take_
