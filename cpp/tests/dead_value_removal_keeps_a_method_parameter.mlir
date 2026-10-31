// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -remove-dead-values | FileCheck %s

// A member function's parameters are its trait's contract: dead-value removal
// keeps a parameter its body never reads, since the impl's member list names
// the function beside its calls.

// CHECK: func.func private @A_i32.m(%{{.*}}: !trait.claim<@A[i32]>, %{{.*}}: i32, %{{.*}}: i32) -> i32 {

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.m]
func.func private @A.m(!trait.claim<@A[!T]>, !T, i32) -> i32
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.m -> @A_i32.m] {
}
func.func private @A_i32.m(%self: !trait.claim<@A[i32]>, %x: i32, %unused: i32) -> i32 {
  return %x : i32
}
func.func @main(%x: i32) -> i32 {
  %c = trait.witness @A_i32 for @A[i32]
  %r = trait.func.call @A.m(%c, %x, %x) : (!trait.claim<@A[i32] by @A_i32>, i32, i32) -> i32
  return %r : i32
}
