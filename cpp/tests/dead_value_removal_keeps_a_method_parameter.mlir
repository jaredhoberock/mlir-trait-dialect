// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -remove-dead-values | FileCheck %s

// A method's parameters are its trait's contract: dead-value removal keeps a
// parameter its body never reads.

// CHECK: trait.method @m(%{{.*}}: i32, %{{.*}}: i32) -> i32 {

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
  trait.method @m(!T, i32) -> i32
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @m(%x: i32, %unused: i32) -> i32 {
    trait.return %x : i32
  }
}
func.func @main(%x: i32) -> i32 {
  %c = trait.witness @A_i32 for @A[i32]
  %r = trait.method.call %c @A[i32]::@m(%x, %x) : (i32, i32) -> i32 by @A_i32
  return %r : i32
}
