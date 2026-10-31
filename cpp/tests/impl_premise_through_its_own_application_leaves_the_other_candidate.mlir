// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// @I's premise projects through the application selection is choosing an impl
// for, so @I is no candidate there (its evidence would cite the proof it is a
// premise of), and the unconditional impl beside it is the one candidate:
// selection chooses it rather than counting both. A premise projection's
// associated-type argument is judged before any binding is substituted for
// the projection, so the argument naming no type refuses @I too.

!T = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!T]>) assoc {Item = 0}
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[!T]>,
    %e: !trait.claim<!trait.proj<@A[!T], "Item", [!trait.proj<@B[!T], "Item">]> = i64>) bindings {Item = i64} {
}
trait.impl private @J(%s: !trait.claim<@A[i32]>) bindings {Item = i64} {
}
// A function with a body taking the claim is a template, cloned at the proof
// its call supplies.
func.func private @need(%a0: !trait.claim<@A[i32]>) {
  return
}
// CHECK-LABEL: func.func @run()
// CHECK: trait.witness @J for @A[i32]
func.func @run() {
  %e = trait.allege @A[i32]
  trait.func.call @need(%e) : (!trait.claim<@A[i32]>) -> ()
  return
}

// -----

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 0}
trait.impl private @I(%s: !trait.claim<@A[!T]>, %e: !trait.claim<!trait.proj<@A[!T], "Item"> = i64>) bindings {Item = i64} {
}
trait.impl private @J(%s: !trait.claim<@A[i32]>) bindings {Item = i64} {
}
// A function with a body taking the claim is a template, cloned at the proof
// its call supplies.
func.func private @need(%a0: !trait.claim<@A[i32]>) {
  return
}
// CHECK-LABEL: func.func @run()
// CHECK: trait.witness @J for @A[i32]
func.func @run() {
  %e = trait.allege @A[i32]
  trait.func.call @need(%e) : (!trait.claim<@A[i32]>) -> ()
  return
}
