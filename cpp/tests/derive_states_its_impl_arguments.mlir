// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=VERIFIED
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A derive states the argument the impl's parameter takes and one premise per
// entry of the impl's where clause, the equality entry included: the derived
// application is the header at that argument and each premise the entry
// there, a substitution the verifier checks with no reading. Monomorphized,
// the derive becomes a witness of the proof through the impl it cites.

// VERIFIED: trait.derive @Tr[tuple<!trait.poly<0>>] from @Tr_tuple[!trait.poly<1> = !trait.poly<0>] given(%arg0, %arg1)

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Tr[!T] {
  trait.assoc_type @Out
  func.func private @get(!T) -> i64
}

trait.impl private @Tr_i32 for @Tr[i32] {
  trait.assoc_type @Out = i64
  func.func @get(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

trait.impl private @Tr_tuple for @Tr[tuple<!U>] where [@Tr[!U], !trait.proj<@Tr[!U], "Out"> = i64] {
  trait.assoc_type @Out = i64
  func.func @get(%x: tuple<!U>) -> i64 {
    %c = arith.constant 35 : i64
    return %c : i64
  }
}

func.func private @g(%t: !trait.claim<@Tr[!T]>, %e: !trait.claim<!trait.proj<@Tr[!T], "Out"> = i64>, %x: tuple<!T>) -> i64 {
  %d = trait.derive @Tr[tuple<!T>] from @Tr_tuple[!U = !T] given(%t, %e) : (!trait.claim<@Tr[!T]>, !trait.claim<!trait.proj<@Tr[!T], "Out"> = i64>)
  %r = trait.method.call %d @Tr[tuple<!T>]::@get(%x) : (tuple<!T>) -> i64
  return %r : i64
}

func.func @main(%x: tuple<i32>) -> i64 {
  %t = trait.allege @Tr[i32]
  %e = trait.witness proj_resolve !trait.proj<@Tr[i32], "Out"> resolves i64 by @Tr_i32
    : !trait.claim<!trait.proj<@Tr[i32], "Out"> = i64>
  %r = trait.func.call @g(%t, %e, %x) : (!trait.claim<@Tr[i32]>, !trait.claim<!trait.proj<@Tr[i32], "Out"> = i64>, tuple<i32>) -> i64
  return %r : i64
}

// CHECK: func.func private @[[GET:Tr_tuple_[_a-z0-9]*get[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK: func.func private @[[G:g_[_a-z0-9]*]](%{{.*}}: tuple<i32>) -> i64
// CHECK-NEXT: call @[[GET]](
// CHECK: func.func @main
// CHECK: call @[[G]](
// CHECK-NOT: trait.
