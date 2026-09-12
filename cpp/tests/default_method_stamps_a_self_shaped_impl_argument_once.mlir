// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// The trait's self parameter and the impl's own parameter are both label 0 --
// every trait's self is, and an impl spells its parameters from label 0 up -- so
// the substitution that carries the default method into the impl maps label 0 to
// a spelling that mentions label 0. Stamping it once reads the impl's argument
// as the term the impl supplied and stops; reading the stamped term again as
// though it were the trait's own spelling wraps another tuple around it every
// time, so the clone here is one tuple deep and no deeper.

trait.trait private @Tr[!trait.poly<0>] {
  func.func nested @method(%self: !trait.poly<0>) -> !trait.poly<0> {
    return %self : !trait.poly<0>
  }
}

trait.impl private @Tr_impl for @Tr[tuple<!trait.poly<0>>] {
}

func.func @main() -> i32 {
  %c = arith.constant 0 : i32
  %v = builtin.unrealized_conversion_cast %c : i32 to tuple<i32>
  %w = trait.allege @Tr[tuple<i32>]
  %r = trait.method.call %w @Tr[tuple<i32>]::@method(%v)
    : (tuple<i32>) -> tuple<i32>
  %o = builtin.unrealized_conversion_cast %r : tuple<i32> to i32
  return %o : i32
}

// CHECK-LABEL: func.func private @Tr_impl
// CHECK-SAME: (%{{.*}}: tuple<i32>) -> tuple<i32>
// CHECK: return %{{.*}} : tuple<i32>
// CHECK-LABEL: func.func @main()
// CHECK: call @Tr_impl
// CHECK-SAME: (tuple<i32>) -> tuple<i32>
// CHECK-NOT: tuple<tuple
