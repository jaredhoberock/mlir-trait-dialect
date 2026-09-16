// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A trait.method.call says nothing about the method's own type variables: the
// trait's arguments ride in the receiver claim, and V is read off the argument
// position it stands in. The verifier checks the actuals are the declaration
// instantiated at V := i32.
//
// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Tr[!trait.poly<0>] {
  func.func private @m(!trait.poly<0>, !trait.poly<9>) -> !trait.poly<9>
}

// CHECK: trait.method.call {{.*}}::@m
func.func @caller(%x: i64, %y: i32) -> i32 {
  %c = trait.allege @Tr[i64]
  %r = trait.method.call %c @Tr[i64]::@m(%x, %y) : (i64, i32) -> i32
  return %r : i32
}
