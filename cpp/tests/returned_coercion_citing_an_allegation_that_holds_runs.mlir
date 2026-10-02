// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The control of invalid_returned_coercion_carries_the_allegation_it_cites:
// with @A[i8]::Out bound to i64 the allegation the returned coercion cites
// holds, and the program runs.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> { trait.assoc_type @Out }
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
 trait.assoc_type @Out = i64
 %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
 %r = trait.witness refl : !trait.claim<i64 = i64>
 %c = trait.coerce %r : !trait.claim<i64 = i64> to !trait.claim<!trait.proj<@A[i8], "Out"> = i64> via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
 trait.return %c : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
 %a = trait.witness @I for @A[i8]
 %e = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
 %n = arith.constant 9 : i64
 %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
 return %r : i64
}
