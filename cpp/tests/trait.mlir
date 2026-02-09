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

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait @Add
// CHECK: func.func private @add(!trait.poly<0>, !trait.poly<0>) -> !trait.poly<0>

!AddSelf = !trait.poly<0>
trait.trait @Add[!AddSelf] {
  func.func private @add(!AddSelf, !AddSelf) -> !AddSelf
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait @PartialEq
// CHECK: func.func private @eq(!trait.poly<1>, !trait.poly<2>) -> i1
// CHECK: func.func @neq(%{{.*}}: !trait.poly<1>, %{{.*}}: !trait.poly<2>) -> i1

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
trait.trait @PartialEq[!PartialEqSelf, !PartialEqOther] {
  func.func private @eq(!PartialEqSelf, !PartialEqOther) -> i1
  
  func.func @neq(%self: !PartialEqSelf, %other: !PartialEqOther) -> i1 {
    %partial_eq = trait.assume @PartialEq[!PartialEqSelf, !PartialEqOther]

    %eq = trait.method.call %partial_eq @PartialEq[!PartialEqSelf,!PartialEqOther]::@eq(%self, %other)
      : (!PartialEqSelf, !PartialEqOther) -> i1

    %true = arith.constant true
    %res = arith.xori %eq, %true : i1
    return %res : i1
  }
}

// ---- Test 2: PartialOrd

// CHECK-LABEL: trait @PartialOrd
// CHECK: func.func private @partial_cmp(!trait.poly<3>, !trait.poly<4>) -> !llvm.struct<"ordering", ()>
// CHECK: func.func @lt(%{{.*}}: !trait.poly<3>, %{{.*}}: !trait.poly<4>) -> i1

!ordering = !llvm.struct<"ordering", ()>
!PartialOrdSelf = !trait.poly<3>
!PartialOrdOther = !trait.poly<4>
trait.trait @PartialOrd[!PartialOrdSelf, !PartialOrdOther] where [
  @PartialEq[!PartialOrdSelf, !PartialOrdOther]
]
{
  func.func private @partial_cmp(!PartialOrdSelf, !PartialOrdOther) -> !ordering

  func.func @lt(%self: !PartialOrdSelf, %other: !PartialOrdOther) -> i1 {
    %partial_ord = trait.assume @PartialOrd[!PartialOrdSelf,!PartialOrdOther]

    %cmp = trait.method.call %partial_ord @PartialOrd[!PartialOrdSelf,!PartialOrdOther]::@partial_cmp(%self, %other)
      : (!PartialOrdSelf, !PartialOrdOther) -> !ordering

    %res = arith.constant false
    return %res : i1
  }
}
