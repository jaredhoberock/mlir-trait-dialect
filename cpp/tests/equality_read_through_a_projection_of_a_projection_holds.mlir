// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The control of
// invalid_false_returned_equality_read_through_a_projection_of_a_projection_is_decided:
// with @A[i8]::Out bound to i64 the allegation the inner projection is replaced
// by holds, and the program runs.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
!E = !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.trait private @B(%self: !trait.claim<@B[!T]>) -> !trait.claim<@A[!T]> {}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i64
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !E
}
trait.impl private @J(%self: !trait.claim<@B[i8]>) {
  %w = trait.witness @I for @A[i8]
  trait.return %w : !trait.claim<@A[i8] by @I>
}
func.func @main() -> i64 {
  %b = trait.witness @J for @B[i8]
  %a = trait.project %b[0] : !trait.claim<@B[i8] by @J> -> !trait.claim<@A[i8]>
  %e = trait.project %a[0] : !trait.claim<@A[i8]> -> !E
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!E)
  return %r : i64
}
