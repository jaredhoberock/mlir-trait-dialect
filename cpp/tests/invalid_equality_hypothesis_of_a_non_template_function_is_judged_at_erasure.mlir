// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A function that is no template assumes the equality i64 = i32 as a parameter.
// No evidence can supply that hypothesis, and erasure would remove it with the
// parameter, leaving a body compiled on it; so erasure's entry judges it where
// it is assumed, as it judges an equality an op produces, and refuses it.
// Instantiation has nothing to refuse in these functions, so the refusal is the
// same whether it ran first or not.

// The hypothesis returned beside a value the function computes.
// CHECK: error: equality '!trait.claim<i64 = i32>' does not settle before erasure: its sides resolve to 'i64' and 'i32'
func.func @use(%h: !trait.claim<i64 = i32>) -> (i64, !trait.claim<i64 = i32>) {
  %v = arith.constant 7 : i64
  return %v, %h : i64, !trait.claim<i64 = i32>
}

// -----

// The hypothesis alone, returned as the function's result.
// CHECK: error: equality '!trait.claim<i64 = i32>' does not settle before erasure: its sides resolve to 'i64' and 'i32'
func.func @forward(%h: !trait.claim<i64 = i32>) -> !trait.claim<i64 = i32> {
  return %h : !trait.claim<i64 = i32>
}

// -----

// The hypothesis forwarded to a later block, which assumes it too: each block
// that takes it is judged.
// CHECK: error: equality '!trait.claim<i64 = i32>' does not settle before erasure: its sides resolve to 'i64' and 'i32'
// CHECK: error: equality '!trait.claim<i64 = i32>' does not settle before erasure: its sides resolve to 'i64' and 'i32'
func.func @branch(%h: !trait.claim<i64 = i32>) -> !trait.claim<i64 = i32> {
  cf.br ^next(%h : !trait.claim<i64 = i32>)
^next(%a: !trait.claim<i64 = i32>):
  return %a : !trait.claim<i64 = i32>
}
