// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Verifies that trait.derive parses and prints correctly (roundtrip).

!T0 = !trait.poly<0>

// CHECK: trait.trait @Trait
trait.trait @Trait [!T0] {
  func.func private @method(!T0) -> i32
}

// An unconditional base impl for i32
// CHECK: trait.impl @Trait_impl_i32 for @Trait[i32]
trait.impl @Trait_impl_i32 for @Trait[i32] {
  func.func @method(%self: i32) -> i32 {
    %res = arith.constant 42 : i32
    return %res : i32
  }
}

// A conditional impl: for any U where Trait[U], Trait holds for tuple<U>
!T1 = !trait.poly<1>
// CHECK: trait.impl @Trait_impl_tuple for @Trait[tuple<!trait.poly<1>>]where [@Trait[!trait.poly<1>]]
trait.impl @Trait_impl_tuple for @Trait[tuple<!T1>] where [@Trait[!T1]] {
  func.func @method(%self: tuple<!T1>) -> i32 {
    %a = trait.assume @Trait[!T1]
    %res = arith.constant 1 : i32
    return %res : i32
  }
}

!T2 = !trait.poly<2>

// A polymorphic function that uses trait.derive
// CHECK-LABEL: func.func @poly_fn
func.func @poly_fn(%arg: tuple<!T2>, %t_claim: !trait.claim<@Trait[!T2]>) -> i32 {
  // CHECK: trait.derive @Trait[tuple<!trait.poly<2>>] from @Trait_impl_tuple given(%{{.*}}) : (!trait.claim<@Trait[!trait.poly<2>]>)
  %d = trait.derive @Trait[tuple<!T2>] from @Trait_impl_tuple given(%t_claim) : (!trait.claim<@Trait[!T2]>)
  %res = trait.method.call %d @Trait[tuple<!T2>]::@method(%arg)
    : (tuple<!T2>) -> i32
  return %res : i32
}
