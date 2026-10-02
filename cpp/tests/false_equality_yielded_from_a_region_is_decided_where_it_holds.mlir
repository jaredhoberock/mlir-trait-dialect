// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The control of invalid_false_returned_equality_yielded_from_a_region_is_decided:
// with @A[i8]::Out bound to i64 the allegation the projection yielded from the
// region is replaced by holds, and the program runs.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i64
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i8]
  %e = scf.execute_region -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64> {
    %p = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
    scf.yield %p : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
  }
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
  return %r : i64
}
