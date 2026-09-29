// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The counterpart of the refused derive: inside a template, the derive supplies
// @Vector_blanket's premise Tensor[T]::Shape = i64 as an allegation over the
// template's own variable, unsafe, which the template cannot decide, and the
// derive stands. The instance decides it: cloned at i8, where @Tensor_i8 binds
// Shape to i64, the allegation is proved and the clone stands.

trait.trait private @Tensor[!trait.poly<0>] {
  trait.assoc_type @Shape
}
trait.trait private @Vector[!trait.poly<0>] {}
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {}
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = i64
}

func.func private @needs(%v: !trait.claim<@Vector[!trait.poly<0>]>) {
  return
}

func.func private @f(%t: !trait.claim<@Tensor[!trait.poly<0>]>) {
  %e = trait.allege !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64 unsafe
  %v = trait.derive @Vector[!trait.poly<0>] from @Vector_blanket[!trait.poly<0> = !trait.poly<0>] given(%t, %e)
    : (!trait.claim<@Tensor[!trait.poly<0>]>, !trait.claim<!trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64>)
  trait.func.call @needs(%v) : (!trait.claim<@Vector[!trait.poly<0>]>) -> ()
  return
}

// CHECK: func.func private @[[NEEDS:needs_h[0-9a-f]+]]()
// CHECK: func.func private @[[F:f_h[0-9a-f]+]]()
// CHECK: call @[[NEEDS]]
// CHECK: func.func @main
// CHECK: call @[[F]]
func.func @main() {
  %t = trait.allege @Tensor[i8]
  trait.func.call @f(%t) : (!trait.claim<@Tensor[i8]>) -> ()
  return
}
