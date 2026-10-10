// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An impl's evidence method for a quantified requirement alleges the
// requirement at its binding, and the use that calls the method proves the
// allegation at the method's instance by selection: the impl of `@Rule` at the
// binding's type serves the method call.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>
!B = !trait.poly<4>

trait.trait private @Rule(%self: !trait.claim<@Rule[!S]>) {
  trait.method @size(!S) -> i64
}
trait.impl private @Rule_pair(%self: !trait.claim<@Rule[tuple<i64, i64>]>) {
  trait.method @size(%x: tuple<i64, i64>) -> i64 {
    %c = arith.constant 2 : i64
    trait.return %c : i64
  }
}
trait.trait private @Holds(%self: !trait.claim<@Holds[!S]>) {
  trait.assoc_type @C<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[!trait.poly<0>], "C", [!trait.poly<1>]>]>
}
trait.impl private @Holds_i32(%self: !trait.claim<@Holds[i32]>) {
  trait.assoc_type @C<[!trait.poly<0>]> = tuple<i64, i64>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]> {
    %r = trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]
    trait.return %r : !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]>
  }
}

func.func private @use_rule(%m: !trait.claim<@Rule[!trait.poly<0>]>, %x: !trait.poly<0>) -> i64 {
  %r = trait.method.call %m @Rule[!trait.poly<0>]::@size(%x) : (!trait.poly<0>) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Holds[!trait.poly<0>]>, %x: !trait.proj<@Holds[!trait.poly<0>], "C", [i1]>) -> i64 {
  %m = trait.method.call %h @Holds[!trait.poly<0>]::@requirement_0() : () -> !trait.claim<@Rule[!trait.proj<@Holds[!trait.poly<0>], "C", [i1]>]>
  %r = trait.func.call @use_rule(%m, %x) : (!trait.claim<@Rule[!trait.proj<@Holds[!trait.poly<0>], "C", [i1]>]>, !trait.proj<@Holds[!trait.poly<0>], "C", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Holds[i32], "C", [i1]>) -> i64 {
  %h = trait.allege @Holds[i32]
  %r = trait.func.call @f(%h, %x) : (!trait.claim<@Holds[i32]>, !trait.proj<@Holds[i32], "C", [i1]>) -> i64
  return %r : i64
}

// CHECK-DAG: func.func private @[[SIZE:Rule_pair_h[0-9a-f]+_size]](%{{.*}}: tuple<i64, i64>) -> i64
// CHECK: func.func @main(%{{.*}}: tuple<i64, i64>) -> i64
// CHECK-NOT: trait.
