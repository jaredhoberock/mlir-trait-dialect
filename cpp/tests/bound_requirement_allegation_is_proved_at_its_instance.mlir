// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An impl alleges its bound requirement, and the use that projects it proves
// the instance by selection: the impl of `@Rule` at the binding's type serves
// the method call.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Rule[!S] {
  func.func private @size(!S) -> i64
}
trait.impl private @Rule_pair for @Rule[tuple<i64, i64>] {
  func.func private @size(%x: tuple<i64, i64>) -> i64 {
    %c = arith.constant 2 : i64
    return %c : i64
  }
}
trait.trait private @Holds[!S] where [forall [!trait.bound<0>] -> @Rule[!trait.proj<@Holds[!S], "C", [!trait.bound<0>]>]] {
  trait.assoc_type @C<[!X]>
}
trait.impl private @Holds_i32 for @Holds[i32]
    witnesses [#trait<witness requirement 0 by allege @Rule[tuple<i64, i64>]>] {
  trait.assoc_type @C<[!X]> = tuple<i64, i64>
}

func.func private @use_rule(%m: !trait.claim<@Rule[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Rule[!M]::@size(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Holds[!T]>, %x: !trait.proj<@Holds[!T], "C", [i1]>) -> i64 {
  %m = trait.project %h[0] for [i1] : !trait.claim<@Holds[!T]> -> !trait.claim<@Rule[!trait.proj<@Holds[!T], "C", [i1]>]>
  %r = trait.func.call @use_rule(%m, %x) : (!trait.claim<@Rule[!trait.proj<@Holds[!T], "C", [i1]>]>, !trait.proj<@Holds[!T], "C", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Holds[i32], "C", [i1]>) -> i64 {
  %h = trait.allege @Holds[i32]
  %r = trait.func.call @f(%h, %x) : (!trait.claim<@Holds[i32]>, !trait.proj<@Holds[i32], "C", [i1]>) -> i64
  return %r : i64
}

// CHECK-DAG: func.func private @[[SIZE:Rule_pair_size[_a-z0-9]*]](%{{.*}}: tuple<i64, i64>) -> i64
// CHECK: func.func @main(%{{.*}}: tuple<i64, i64>) -> i64
// CHECK-NOT: trait.
