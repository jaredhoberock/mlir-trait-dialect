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

!A = !trait.poly<0>
// CHECK-NOT: trait.trait @A
trait.trait @A[!A] {}

!Ai = !trait.poly<1>
// CHECK-NOT: trait.impl @A_impl
trait.impl @A_impl for @A[!Ai] {}

!B = !trait.poly<2>
// CHECK-NOT: trait.trait @B
trait.trait @B[!B] {}

!Bi = !trait.poly<3>
// CHECK-NOT: trait.impl @B_impl
trait.impl @B_impl for @B[!Bi] {}

!C = !trait.poly<4>
// CHECK-NOT: trait.trait @C
trait.trait @C[!C] where [
  @A[!C]
] {
  func.func @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK-NOT: trait.impl @C_impl
trait.impl @C_impl for @C[!Ci] where [
  @B[!Ci]
] {}

// CHECK-LABEL: func.func @foo
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @foo(%x: i8) -> i1 {
  %c = trait.allege @C[i8]
  // CHECK: call @C_impl_{{.*}}_method
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// CHECK-NOT: trait.proof
