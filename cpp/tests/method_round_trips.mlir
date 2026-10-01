// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

// A method prints as a func.func does, with no visibility: a required method
// has no body, a bodied one ends in trait.return, and a body may hold several
// blocks. The custom form, the generic form and bytecode all read back to the
// same declarations.

!T = !trait.poly<0>
trait.trait private @Tr[!T] {
  trait.method @required(!T) -> i64
  trait.method @default(%x: !T) -> i64 {
    %s = trait.assume self : !trait.claim<@Tr[!T]>
    %r = trait.method.call %s @Tr[!T]::@required(%x) : (!T) -> i64
    trait.return %r : i64
  }
}

trait.impl private @Tr_i64 for @Tr[i64] {
  trait.method @required(%x: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %x, %zero : i64
    cf.cond_br %negative, ^below, ^above
  ^below:
    trait.return %zero : i64
  ^above:
    trait.return %x : i64
  }
}

// CHECK-LABEL: trait.trait private @Tr
// CHECK-NEXT:    trait.method @required(!trait.poly<0>) -> i64{{$}}
// CHECK-NEXT:    trait.method @default(%{{.*}}: !trait.poly<0>) -> i64 {
// CHECK:           trait.return %{{.*}} : i64

// CHECK-LABEL: trait.impl private @Tr_i64
// CHECK-NEXT:    trait.method @required(%{{.*}}: i64) -> i64 {
// CHECK:           cf.cond_br
// CHECK:           trait.return %{{.*}} : i64
// CHECK:           trait.return %{{.*}} : i64
