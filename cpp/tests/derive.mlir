// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Verifies that trait.derive parses and prints correctly (roundtrip).

!T0 = !trait.poly<0>

// CHECK: trait.trait private @Trait
trait.trait private @Trait(%self: !trait.claim<@Trait[!T0]>) {
  trait.method @method(!T0) -> i32
}

// An unconditional base impl for i32
// CHECK: trait.impl private @Trait_impl_i32(%self: !trait.claim<@Trait[i32]>
trait.impl private @Trait_impl_i32(%self_claim: !trait.claim<@Trait[i32]>) {
  trait.method @method(%self: i32) -> i32 {
    %res = arith.constant 42 : i32
    trait.return %res : i32
  }
}

// A conditional impl: for any U where Trait[U], Trait holds for tuple<U>
!T1 = !trait.poly<1>
// CHECK: trait.impl private @Trait_impl_tuple(%self: !trait.claim<@Trait[tuple<!trait.poly<0>>]>
trait.impl private @Trait_impl_tuple(%self_claim: !trait.claim<@Trait[tuple<!trait.poly<0>>]>, %trait: !trait.claim<@Trait[!trait.poly<0>]>) {
  trait.method @method(%self: tuple<!trait.poly<0>>) -> i32 {
    %res = arith.constant 1 : i32
    trait.return %res : i32
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
