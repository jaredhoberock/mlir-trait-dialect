// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=IR

// A claim nested inside another claim must receive its proof throughout
// specialization. The concrete Hold method and its generic caller must lower
// with no claim types or unrealized conversion casts left behind.

!T = !trait.poly<0>

trait.trait private @Ground[!T] {}

trait.impl private @Ground_all for @Ground[!T] {}

trait.trait private @Hold[!T] {
  func.func private @held() -> !T
}

trait.impl private @Hold_claim for @Hold[!trait.claim<@Ground[i32]>] where [
  @Ground[i32]
] {
  func.func @held() -> !trait.claim<@Ground[i32]> {
    %g = trait.assume @Ground[i32]
    return %g : !trait.claim<@Ground[i32]>
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

// CHECK-LABEL: func.func private @Hold_claim_held()
// CHECK: return
// CHECK-LABEL: func.func private @take_
// CHECK: call @Hold_claim_held() : () -> ()
// CHECK-LABEL: func.func @test()
// CHECK: call @take_
