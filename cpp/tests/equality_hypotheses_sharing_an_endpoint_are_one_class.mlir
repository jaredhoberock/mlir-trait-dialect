// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Equalities chain. @caller's where clause holds @B[!T]::Item = @A[!T]::Item and
// @B[!T]::Item = @C[!T]::Item, so all three projections name one type and the
// callee's @A-spelled result is the @C-spelled result the call site names.
// Nothing else reconciles the two: the call carries no equality operand, and
// neither hypothesis relates @A[!T]::Item to @C[!T]::Item by itself. Read as
// directed rules the two hypotheses share a left endpoint and one displaces the
// other, leaving the chain broken; read as classes they join three members into
// one class.

// CHECK-LABEL: func.func @caller
// CHECK: trait.func.call @callee

!T = !trait.poly<0>
trait.trait private @A[!T] {
  trait.assoc_type @Item
}
trait.trait private @B[!T] {
  trait.assoc_type @Item
}
trait.trait private @C[!T] {
  trait.assoc_type @Item
}

!X = !trait.poly<1>
func.func private @callee(!X, !trait.claim<@A[!X]>) -> !trait.proj<@A[!X], "Item">

!Y = !trait.poly<2>
func.func @caller(%x: !Y, %a: !trait.claim<@A[!Y]>,
    %toA: !trait.claim<!trait.proj<@B[!Y], "Item"> = !trait.proj<@A[!Y], "Item">>,
    %toC: !trait.claim<!trait.proj<@B[!Y], "Item"> = !trait.proj<@C[!Y], "Item">>)
    -> !trait.proj<@C[!Y], "Item"> {
  %r = trait.func.call @callee(%x, %a)
    : (!Y, !trait.claim<@A[!Y]>) -> !trait.proj<@C[!Y], "Item">
  return %r : !trait.proj<@C[!Y], "Item">
}
