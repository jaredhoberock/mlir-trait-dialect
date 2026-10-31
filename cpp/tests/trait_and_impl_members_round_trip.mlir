// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

// A trait lists its members in declaration order, a member with a default
// marked, and an impl pairs each member with the function implementing it.
// The members are module-level functions: a required one has no body, a
// default and an impl's function have one, which may have several blocks. The
// custom form, the generic form and bytecode all read back to the same
// declarations.

!T = !trait.poly<0>
trait.trait private @Tr(!trait.claim<@Tr[!T]>) members [@Tr.required, default @Tr.default]
func.func private @Tr.required(!trait.claim<@Tr[!T]>, !T) -> i64
func.func private @Tr.default(%self: !trait.claim<@Tr[!T]>, %x: !T) -> i64 {
  %r = trait.func.call @Tr.required(%self, %x) : (!trait.claim<@Tr[!T]>, !T) -> i64
  return %r : i64
}

trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) members [@Tr.required -> @Tr_i64.required, @Tr.default -> @Tr.default] {
}
func.func private @Tr_i64.required(%self: !trait.claim<@Tr[i64]>, %x: i64) -> i64 {
  %zero = arith.constant 0 : i64
  %negative = arith.cmpi slt, %x, %zero : i64
  cf.cond_br %negative, ^below, ^above
^below:
  return %zero : i64
^above:
  return %x : i64
}

// CHECK-LABEL: trait.trait private @Tr(!trait.claim<@Tr[!trait.poly<0>]>){{$}}
// CHECK-NEXT: members [
// CHECK-NEXT: @Tr.required,
// CHECK-NEXT: default @Tr.default{{$}}
// CHECK-NEXT: ]{{$}}
// CHECK-NEXT:    func.func private @Tr.required(!trait.claim<@Tr[!trait.poly<0>]>, !trait.poly<0>) -> i64{{$}}
// CHECK-NEXT:    func.func private @Tr.default(%{{.*}}: !trait.claim<@Tr[!trait.poly<0>]>, %{{.*}}: !trait.poly<0>) -> i64 {
// CHECK:           return %{{.*}} : i64

// CHECK-LABEL: trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>){{$}}
// CHECK-NEXT: members [
// CHECK-NEXT: @Tr.required -> @Tr_i64.required,
// CHECK-NEXT: @Tr.default -> @Tr.default{{$}}
// CHECK-NEXT: ] {
// CHECK:         func.func private @Tr_i64.required(%{{.*}}: !trait.claim<@Tr[i64]>, %{{.*}}: i64) -> i64 {
// CHECK:           cf.cond_br
// CHECK:           return %{{.*}} : i64
// CHECK:           return %{{.*}} : i64
