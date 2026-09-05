// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A trait.method.call carries the method's own type arguments explicitly (the
// trait's arguments ride in the receiver claim). The verifier reads the binding
// V := i32 from the call and checks the actuals are an instance of it, and the
// two parallel arrays round-trip through the attribute dictionary.
//
// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Tr[!trait.poly<0>] {
  func.func private @m(!trait.poly<0>, !trait.poly<9>) -> !trait.poly<9>
}

// CHECK: trait.method.call {{.*}}::@m
// CHECK: type_args = [i32], type_params = [!trait.poly<9>]
func.func @caller(%x: i64, %y: i32) -> i32 {
  %c = trait.allege @Tr[i64]
  %r = trait.method.call %c @Tr[i64]::@m(%x, %y) : (i64, i32) -> i32 attributes {type_params = [!trait.poly<9>], type_args = [i32]}
  return %r : i32
}
