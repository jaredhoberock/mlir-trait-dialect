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
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Item }
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[!T]>,
    %e: !trait.claim<!trait.proj<@A[!T], "Item", [!trait.proj<@B[!T], "Item">]> = i64>) {
  trait.assoc_type @Item<[!trait.poly<1>]> = i64
}
trait.impl private @J(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item<[!trait.poly<0>]> = i64 }
func.func private @need(!trait.claim<@A[i32]>)
// CHECK-LABEL: func.func @run()
// CHECK: trait.witness @J for @A[i32]
func.func @run() {
  %e = trait.allege @A[i32]
  func.call @need(%e) : (!trait.claim<@A[i32]>) -> ()
  return
}

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.impl private @I(%s: !trait.claim<@A[!T]>, %e: !trait.claim<!trait.proj<@A[!T], "Item"> = i64>) { trait.assoc_type @Item = i64 }
trait.impl private @J(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item = i64 }
func.func private @need(!trait.claim<@A[i32]>)
// CHECK-LABEL: func.func @run()
// CHECK: trait.witness @J for @A[i32]
func.func @run() {
  %e = trait.allege @A[i32]
  func.call @need(%e) : (!trait.claim<@A[i32]>) -> ()
  return
}
