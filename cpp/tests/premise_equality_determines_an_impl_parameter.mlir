// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @Fold_gen binds !Acc nowhere in its header: the only thing that says what
// !Acc is, is its where-clause equality. Discharging that premise at selection
// is what assigns it -- @Out[i32]::Output resolves to i64, so !Acc is i64 --
// and only then is the candidate's argument list complete enough to specialize
// its associated-type binding and its method.

!F = !trait.poly<0>

trait.trait private @Out[!F] {
  trait.assoc_type @Output
}

trait.impl private @Out_i32 for @Out[i32] {
  trait.assoc_type @Output = i64
}

!G = !trait.poly<1>
trait.trait private @Fold[!G] {
  trait.assoc_type @Sum
  func.func private @run(!G) -> !trait.proj<@Fold[!G], "Sum">
}

!Acc = !trait.poly<2>
trait.impl private @Fold_gen for @Fold[!G] where [!trait.proj<@Out[!G], "Output"> = !Acc] {
  trait.assoc_type @Sum = !Acc
  func.func @run(%x: !G) -> !Acc {
    %r = builtin.unrealized_conversion_cast %x : !G to !Acc
    return %r : !Acc
  }
}

// The instance returns i64, which only the premise determined.
// CHECK-LABEL: func.func @main
// CHECK: call @Fold_gen
// CHECK-SAME: (i32) -> i64
func.func @main(%x: i32) {
  %c = trait.allege @Fold[i32]
  %r = trait.method.call %c @Fold[i32]::@run(%x)
    : (i32) -> !trait.proj<@Fold[i32], "Sum">
  return
}
