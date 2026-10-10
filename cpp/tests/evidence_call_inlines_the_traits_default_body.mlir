// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Has's quantified requirement @requirement_0 has a default body, a witness of
// @Marker_i64, and @Has_i32 does not define it. A call of it through a proof
// of @Has_i32 is replaced by the trait's default body, as a call of a method an
// impl leaves to its trait is cut from the default, and runs @Marker_i64's
// @mark on 41.

// CHECK: {{^}}42{{$}}

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) { trait.method @mark(i64) -> i64 }
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(%m: !trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]> {
    %r = trait.witness @Marker_i64 for @Marker[i64]
    %eq = trait.allege !trait.proj<@Has[!S], "A", [!X]> = i64
    %c = trait.coerce %r : !trait.claim<@Marker[i64] by @Marker_i64> to !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]> via (%eq) : (!trait.claim<!trait.proj<@Has[!S], "A", [!X]> = i64>)
    trait.return %c : !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
  }
}
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) {
  trait.method @mark(%x: i64) -> i64 { %c = arith.constant 1 : i64 %r = arith.addi %x, %c : i64 trait.return %r : i64 }
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!trait.poly<0>]> = i64
}
func.func @main() -> i64 {
  %x = arith.constant 41 : i64
  %has = trait.witness @Has_i32 for @Has[i32]
  %m = trait.witness @Marker_i64 for @Marker[i64]
  %ev = trait.method.call %has @Has[i32]::@requirement_0(%m) : (!trait.claim<@Marker[i64] by @Marker_i64>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [i64]>]> by @Has_i32
  %r = trait.method.call %ev @Marker[!trait.proj<@Has[i32], "A", [i64]>]::@mark(%x) : (i64) -> i64
  return %r : i64
}
