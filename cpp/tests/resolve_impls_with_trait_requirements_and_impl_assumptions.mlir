// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK: trait.trait private @
trait.trait private @A[!A] {}

!Ai = !trait.poly<1>
// CHECK: trait.impl private @A_impl
trait.impl private @A_impl for @A[!Ai] {}

!B = !trait.poly<2>
// CHECK: trait.trait private @B
trait.trait private @B[!B] {}

!Bi = !trait.poly<3>
// CHECK: trait.impl private @B_impl
trait.impl private @B_impl for @B[!Bi] {}

!C = !trait.poly<4>
// CHECK: trait.trait private @C
trait.trait private @C[!C] where [
  @A[!C]
] {
  func.func @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK: trait.impl private @C_impl
trait.impl private @C_impl for @C[!Ci] where [
  @B[!Ci]
] {}

func.func @foo(%x: i8) -> i1 {
  // CHECK: trait.witness @C_impl_{{.*}}_p for @C[i8]
  %c = trait.allege @C[i8]
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// CHECK: trait.proof private @A_impl_{{.*}}_p proves @A_impl for @A[i8] given []
// CHECK: trait.proof private @B_impl_{{.*}}_p proves @B_impl for @B[i8] given []
// CHECK: trait.proof private @C_impl_{{.*}}_p proves @C_impl for @C[i8] given [@A_impl_{{.*}}_p, @B_impl_{{.*}}_p]
