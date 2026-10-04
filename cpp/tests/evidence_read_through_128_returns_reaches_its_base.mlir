// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Requirement evidence read through one hundred and twenty-eight returns, each
// a projection of the next impl's, to the one-hundred-and-twenty-ninth, which
// returns a witness: the evidence has a base, however long the reading, and
// @Base's method runs.

// CHECK: 7

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.impl private @Base(%s: !trait.claim<@B[i32]>) { trait.method @v() -> i64 { %v = arith.constant 7 : i64 trait.return %v : i64 } }
trait.trait private @A(%s: !trait.claim<@A[!T,!U]>) -> !trait.claim<@B[!U]> {}
// REPEAT 1 128: trait.impl private @I{k}(%s: !trait.claim<@A[i{k},i32]>) { %a = trait.witness @I{k+1} for @A[i{k+1},i32] %b = trait.project %a[0] : !trait.claim<@A[i{k+1},i32] by @I{k+1}> -> !trait.claim<@B[i32]> trait.return %b : !trait.claim<@B[i32]> }
trait.impl private @I129(%s: !trait.claim<@A[i129,i32]>) { %b = trait.witness @Base for @B[i32] trait.return %b : !trait.claim<@B[i32] by @Base> }
func.func @main() -> i64 {
  %a = trait.witness @I1 for @A[i1,i32]
  %b = trait.project %a[0] : !trait.claim<@A[i1,i32] by @I1> -> !trait.claim<@B[i32]>
  %v = trait.method.call %b @B[i32]::@v() : () -> i64
  return %v : i64
}
