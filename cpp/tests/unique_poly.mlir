// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @foo
// CHECK-1: !trait.poly<0>
// CHECK-3: !trait.poly<-1>
// CHECK-NOT: !trait.poly<1>
// CHECK-NOT: !trait.poly<-2>
!T = !trait.poly<0>
!U = !trait.poly<unique>
func.func @foo(%a: !T, %b: !U) -> !U {
  return %b : !U
}
