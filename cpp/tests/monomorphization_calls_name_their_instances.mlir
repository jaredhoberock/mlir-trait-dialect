// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Monomorphizing calls whose claims resolve several applications and their
// premises gives each call its own instance: the two claims on one callee name
// two instances, and every rewritten call names an instance created for it.
// Instances are cut in the order the stage reaches their calls, which is no
// part of what this pins.

!T = !trait.poly<0>

trait.trait private @Zero(%self: !trait.claim<@Zero[!T]>) {}
trait.trait private @One(%self: !trait.claim<@One[!T]>) {}
trait.trait private @Two(%self: !trait.claim<@Two[!T]>) -> (!trait.claim<@Zero[!T]>, !trait.claim<@One[!T]>) {}

trait.impl private @Zero_i32(%self: !trait.claim<@Zero[i32]>) {}
trait.impl private @Zero_i64(%self: !trait.claim<@Zero[i64]>) {}
trait.impl private @One_i32(%self: !trait.claim<@One[i32]>) {}
trait.impl private @Two_i32(%self: !trait.claim<@Two[i32]>) {
  %req0 = trait.allege @Zero[i32]
  %req1 = trait.allege @One[i32]
  trait.return %req0, %req1 : !trait.claim<@Zero[i32]>, !trait.claim<@One[i32]>
}

!P = !trait.poly<1>

func.func private @hold_zero(%zero: !trait.claim<@Zero[!P]>) {
  return
}

func.func private @hold_two(%two: !trait.claim<@Two[!P]>) {
  return
}

func.func @main() {
  %zero64 = trait.allege @Zero[i64]
  %zero32 = trait.allege @Zero[i32]
  %two = trait.allege @Two[i32]
  trait.func.call @hold_zero(%zero32) : (!trait.claim<@Zero[i32]>) -> ()
  trait.func.call @hold_zero(%zero64) : (!trait.claim<@Zero[i64]>) -> ()
  trait.func.call @hold_two(%two) : (!trait.claim<@Two[i32]>) -> ()
  return
}

// CHECK-DAG: func.func private @hold_zero_[[FIRST:h[0-9a-f]+]]()
// CHECK-DAG: func.func private @hold_zero_[[SECOND:h[0-9a-f]+]]()
// CHECK-DAG: func.func private @hold_two_[[THIRD:h[0-9a-f]+]]()
// CHECK-LABEL: func.func @main()
// CHECK-DAG: call @hold_zero_[[FIRST]]() : () -> ()
// CHECK-DAG: call @hold_zero_[[SECOND]]() : () -> ()
// CHECK-DAG: call @hold_two_[[THIRD]]() : () -> ()
