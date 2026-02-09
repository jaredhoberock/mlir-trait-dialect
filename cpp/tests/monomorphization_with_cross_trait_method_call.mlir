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

trait.trait @A[!trait.poly<0>] {
  func.func nested @a(!trait.poly<0>) -> i1
}
trait.impl @A_impl_i1 for @A[i1] {
  func.func nested @a(%arg0: i1) -> i1 {
    return %arg0 : i1
  }
}
trait.trait @B[!trait.poly<1>] {
  func.func nested @b(%arg0: !trait.poly<1>) -> i1
}
trait.impl @B_impl_i1 for @B[i1] {
  func.func nested @b(%arg0: i1) -> i1 {
    %a = trait.allege @A[i1]

    // test that we are able to trait.method.call
    // from inside a method to another trait
    %res = trait.method.call %a @A[i1]::@a(%arg0) : (i1) -> i1
    return %res : i1
  }
}
func.func @test(%arg0 : i1) -> i1 {
  %b = trait.allege @B[i1]
  %res = trait.method.call %b @B[i1]::@b(%arg0) : (i1) -> i1
  return %res : i1
}

// Instance of A::a
// CHECK: func.func private @A_impl_i1_a(

// Instance of B::b that calls A_impl_i1_a
// CHECK: func.func private @B_impl_i1_b(
// CHECK: call @A_impl_i1_a

// Top-level test calls B_impl_i1_b
// CHECK: func.func @test(
// CHECK: call @B_impl_i1_b

// No trait ops should remain
// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.func.call
// CHECK-NOT: trait.allege
