// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(symbol-dce)' | FileCheck %s

// An equality's endpoints are ordinary sub-elements, so the symbol walker sees
// what they name: a private trait named nowhere but inside one is rooted by the
// operation holding the equality and stands through collection. Endpoints
// stand in two positions and both are covered here -- a claim type in a live
// function's signature, and a bare equality attribute riding on that function
// -- and each names a trait nothing else in the module mentions. Stock
// symbol-dce runs alone: nothing of the dialect is special-cased.

trait.trait private @InSignature(%self: !trait.claim<@InSignature[!trait.poly<0>]>) {
  trait.assoc_type @Out
}

trait.trait private @InAttribute(%self: !trait.claim<@InAttribute[!trait.poly<0>]>) {
  trait.assoc_type @Out
}

// CHECK-DAG: trait.trait private @InSignature
// CHECK-DAG: trait.trait private @InAttribute
// CHECK-DAG: func.func @main
func.func @main(%c: !trait.claim<!trait.proj<@InSignature[i32], "Out"> = i64>) -> i32
    attributes {evidence = #trait<equality i64 = !trait.proj<@InAttribute[i32], "Out">>} {
  %z = arith.constant 0 : i32
  return %z : i32
}
