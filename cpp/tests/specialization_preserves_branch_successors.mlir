// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s \
// RUN:   -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' \
// RUN:   | FileCheck %s

!T = !trait.poly<0>
func.func private @f(%x: !T) -> !T {
  cf.br ^next(%x : !T)
^next(%y: !T):
  return %y : !T
}
func.func @root(%x: i32) -> i32 {
  %r = trait.func.call @f(%x)
    {type_params = [!T], type_args = [i32]} : (i32) -> i32
  return %r : i32
}

// CHECK: func.func private @[[F:f_h[0-9a-f]+]](%[[X:.*]]: i32) -> i32 {
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[X]] : i32)
// CHECK: ^[[NEXT]](%[[Y:.*]]: i32):
// CHECK: return %[[Y]] : i32
// CHECK: func.func @root
// CHECK: call @[[F]]
