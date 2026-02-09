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

// this trait returns some type from get
!R = !trait.poly<0>
trait.trait @Get[!R] {
  func.func private @get() -> !R
}

// this trait will be used in an impl where below
!A = !trait.poly<1>
trait.trait @Assumption[!A] {}

// a blanket impl for @Assumption for all types
trait.impl @Assumption_impl for @Assumption[!A] {}

// this impl returns an assumption claim from @get
trait.impl @Get_impl_claim for @Get[!trait.claim<@Assumption[i32]>] where [
  @Assumption[i32]
] {
  func.func @get() -> !trait.claim<@Assumption[i32]> {
    %res = trait.assume @Assumption[i32]
    return %res : !trait.claim<@Assumption[i32]>
  }
}

// this polymorphic function calls get and returns its result
// CHECK-LABEL: func.func @call_get_{{.*}}
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @call_get(%c: !trait.claim<@Get[!R]>) -> !R {
  %res = trait.method.call %c @Get[!R]::@get()
    : () -> !R
  return %res : !R
}

// CHECK-LABEL: func.func @test
// CHECK-NOT: builtin.unrealized_conversion_cast
// test that we can call a polymorphic function that returns a claim
func.func @test() {
  // allege an impl for @Get exists which returns this type of claim
  %a = trait.allege @Get[!trait.claim<@Assumption[i32]>]

  // call a polymorphic function that returns the !trait.claim
  trait.func.call @call_get(%a)
    : (!trait.claim<@Get[!trait.claim<@Assumption[i32]>]>) -> !trait.claim<@Assumption[i32]>

  return
}

// CHECK-NOT: trait.assume
// CHECK-NOT: trait.func.call
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.project
// CHECK-NOT: trait.proof
// CHECK-NOT: trait.trait
