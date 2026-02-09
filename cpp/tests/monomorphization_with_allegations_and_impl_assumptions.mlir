// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T0 = !trait.poly<0>
// CHECK-NOT: trait.trait @A
trait.trait @A [!T0] {
  func.func private @method_a(!T0) -> i32
}

!T1 = !trait.poly<1>
// CHECK-NOT: trait.trait @B
trait.trait @B [!T1] {
  func.func private @method_b(!T1) -> i32
}

// CHECK-NOT: trait.impl for @B[i32]
trait.impl @B_impl_i32 for @B[i32] {
  func.func @method_b(%arg0: i32) -> i32 {
    %res = arith.constant 1 : i32
    return %res : i32
  }
}

// CHECK-NOT: trait.impl for @B[i8]
trait.impl @B_impl_i8 for @B[i8] {
  func.func @method_b(%arg: i8) -> i32 {
    %res = arith.constant 1 : i32
    return %res : i32
  }
}

!T2 = !trait.poly<2>
// CHECK-NOT: trait.impl @A_impl_poly
trait.impl @A_impl_poly for @A[!T2] where [@B[!T2]] {
  func.func @method_a(%arg0: !T2) -> i32 {
    %b = trait.assume @B[!T2]
    %res = trait.method.call %b @B[!T2]::@method_b(%arg0)
      : (!T2) -> i32
    return %res : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @test() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %c7_i8 = arith.constant 7 : i8

  %a_i32 = trait.allege @A[i32]
  // CHECK: call @A_impl_poly_{{.*}}_method_a
  %res0 = trait.method.call %a_i32 @A[i32]::@method_a(%c42_i32)
    : (i32) -> i32

  %a_i8 = trait.allege @A[i8]
  // CHECK: call @A_impl_poly_{{.*}}_method_a
  %res1 = trait.method.call %a_i8 @A[i8]::@method_a(%c7_i8)
    : (i8) -> i32

  %res = arith.addi %res0, %res1 : i32
  return %res : i32
}

// CHECK-NOT: trait.proof
