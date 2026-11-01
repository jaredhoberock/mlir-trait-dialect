// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The twin of the refused copy: the impl's @f spells its own labels for the
// trait's member parameters, one for one. That is a renaming, so a call's type
// arguments -- named under the trait's labels -- rekey onto the clone's.

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.f]
func.func private @T.f(!trait.claim<@T[!trait.poly<0>]>, !trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>

trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@T.f -> @I.f] {
}
func.func private @I.f(%self: !trait.claim<@T[i32]>, %x: i32, %m: !trait.poly<0>, %n: !trait.poly<1>) -> !trait.poly<0> {
  return %m : !trait.poly<0>
}

// CHECK: func.func private @[[F:I.f_h[0-9a-f]+]](%{{.*}}: i32, %{{.*}}: i64, %{{.*}}: i8) -> i64
// CHECK: func.func @main
// CHECK: call @[[F]]
func.func @main(%x: i32, %m: i64, %n: i8) -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.func.call @T.f(%c, %x, %m, %n) : (!trait.claim<@T[i32]>, i32, i64, i8) -> i64
  return %r : i64
}
