// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The twin of the refused copy: the impl's @f spells its own labels for the
// trait's method parameters, one for one. That is a renaming, so a call's type
// arguments -- named under the trait's labels -- rekey onto the clone's.

trait.trait private @T[!trait.poly<0>] {
  func.func private @f(!trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>
}

trait.impl private @I for @T[i32] {
  func.func @f(%x: i32, %m: !trait.poly<7>, %n: !trait.poly<8>) -> !trait.poly<7> {
    return %m : !trait.poly<7>
  }
}

// CHECK: func.func private @[[F:I_f[0-9a-z_]*]](%{{.*}}: i32, %{{.*}}: i64, %{{.*}}: i8) -> i64
// CHECK: func.func @main
// CHECK: call @[[F]]
func.func @main(%x: i32, %m: i64, %n: i8) -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.method.call %c @T[i32]::@f(%x, %m, %n) : (i32, i64, i8) -> i64
    attributes {type_params = [!trait.poly<1>, !trait.poly<2>], type_args = [i64, i8]}
  return %r : i64
}
