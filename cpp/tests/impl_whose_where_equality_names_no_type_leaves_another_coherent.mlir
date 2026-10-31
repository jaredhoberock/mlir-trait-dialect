// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=MONO

// @A_only_when_dead applies where @Dead[T]::Out = @Dead[T]::Out, and no impl of
// @Dead serves i32, so at i32 the premise is about no type and the impl is no
// candidate: @A_i32 is the one impl selection chooses for @A[i32], rather than
// one of two it cannot choose between.

!T = !trait.poly<0>
trait.trait private @Dead(!trait.claim<@Dead[!T]>) assoc {Out = 0}
trait.impl private @Dead_f32(%self: !trait.claim<@Dead[f32]>) bindings {Out = f32} {
}
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 0}
trait.impl private @A_only_when_dead(%self: !trait.claim<@A[!T]>,
                                     %eq: !trait.claim<!trait.proj<@Dead[!T], "Out"> = !trait.proj<@Dead[!T], "Out">>) bindings {Item = f32} {
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) bindings {Item = i64} {
}

// A function with a body taking the claim is a template, cloned at the proof
// its call supplies.
func.func private @need(%a0: !trait.claim<@A[i32]>) {
  return
}

// CHECK-LABEL: func.func @f()
// CHECK: trait.witness @A_i32 for @A[i32]
// MONO-LABEL: func.func @f()
// MONO: call @need_h{{[0-9a-f]+}}()
func.func @f() {
  %e = trait.allege @A[i32]
  trait.func.call @need(%e) : (!trait.claim<@A[i32]>) -> ()
  return
}
