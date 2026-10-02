// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -test-single-fold | FileCheck %s

// The folder alone, with no rewrite pattern: a method is not isolated from
// above, so the folder would place the constants it materializes while folding a
// method's body in the nearest isolated ancestor, the trait or impl, whose child
// list refuses them. The dialect claims a method's body as the region to
// materialize into, so each constant stays in the method that made it: `7 + 1`
// folds to 8 inside @a's execute_region, @b keeps its own 7 although @a also
// spelled one, and the trait's default keeps the 3 it folds to.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) {
  trait.method @a(!T) -> i64
  trait.method @b(!T) -> i64
  trait.method @c(%x: !T) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.addi %one, %two : i64
    trait.return %three : i64
  }
}

trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @a(%x: i64) -> i64 {
    %r = scf.execute_region -> i64 {
      %seven = arith.constant 7 : i64
      %one = arith.constant 1 : i64
      %eight = arith.addi %seven, %one : i64
      %sum = arith.addi %x, %eight : i64
      scf.yield %sum : i64
    }
    trait.return %r : i64
  }
  trait.method @b(%x: i64) -> i64 {
    %seven = arith.constant 7 : i64
    %r = arith.muli %x, %seven : i64
    trait.return %r : i64
  }
}

// CHECK-LABEL: trait.trait private @Tr
// CHECK-NEXT:    trait.method @a
// CHECK-NEXT:    trait.method @b
// CHECK-NEXT:    trait.method @c
// CHECK-NEXT:      arith.constant 3 : i64

// CHECK-LABEL: trait.impl private @Tr_i64
// CHECK-NEXT:    trait.method @a
// CHECK-NOT:     trait.method
// CHECK:           arith.constant 8 : i64
// CHECK:         trait.method @b
// CHECK-NEXT:      arith.constant 7 : i64
// CHECK-NEXT:      arith.muli
// CHECK-NEXT:      trait.return
