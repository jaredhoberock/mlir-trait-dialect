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

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK: trait.trait @
trait.trait @A[!A] {}

!Ai = !trait.poly<1>
// CHECK: trait.impl @A_impl
trait.impl @A_impl for @A[!Ai] {}

!B = !trait.poly<2>
// CHECK: trait.trait @B
trait.trait @B[!B] {}

!Bi = !trait.poly<3>
// CHECK: trait.impl @B_impl
trait.impl @B_impl for @B[!Bi] {}

!C = !trait.poly<4>
// CHECK: trait.trait @C
trait.trait @C[!C] where [
  @A[!C]
] {
  func.func @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK: trait.impl @C_impl
trait.impl @C_impl for @C[!Ci] where [
  @B[!Ci]
] {}

func.func @foo(%x: i8) -> i1 {
  // CHECK: trait.witness @C_impl_{{.*}}_p for @C[i8]
  %c = trait.allege @C[i8]
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// CHECK: trait.proof @A_impl_{{.*}}_p proves @A_impl for @A[i8] given []
// CHECK: trait.proof @B_impl_{{.*}}_p proves @B_impl for @B[i8] given []
// CHECK: trait.proof @C_impl_{{.*}}_p proves @C_impl for @C[i8] given [@A_impl_{{.*}}_p, @B_impl_{{.*}}_p]
