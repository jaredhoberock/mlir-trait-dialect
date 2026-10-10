// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A quantified requirement, `forall X where Marker[X] -> Marker[Has[S]::A<X>]`,
// is a bodiless evidence method of the trait: its parameters are the premise
// claims and its result the conclusion. An impl provides it with a body that
// returns the evidence, and a caller selects it with a method call whose
// result is a claim; monomorphization lowers the call through that claim to
// the impl the evidence names.

// ROUNDTRIP:      trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) {
// ROUNDTRIP-NEXT:   trait.assoc_type @A<[!trait.poly<1>]>
// ROUNDTRIP-NEXT:   trait.method @requirement_0(!trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
// ROUNDTRIP-NEXT: }
// ROUNDTRIP:      trait.method.call %{{.*}} @Has[i32]::@requirement_0(%{{.*}})
// ROUNDTRIP-NEXT:   : (!trait.claim<@Marker[i64] by @Marker_i64>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [i64]>]>

// CHECK: {{^}}42{{$}}

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {
  trait.method @mark(i64) -> i64
}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) {
  trait.method @mark(%x: i64) -> i64 {
    %c = arith.constant 1 : i64
    %r = arith.addi %x, %c : i64
    trait.return %r : i64
  }
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!trait.poly<0>]> = i64
  trait.method @requirement_0(%m: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[i64] by @Marker_i64> {
    %r = trait.witness @Marker_i64 for @Marker[i64]
    trait.return %r : !trait.claim<@Marker[i64] by @Marker_i64>
  }
}
func.func @main() -> i64 {
  %x = arith.constant 41 : i64
  %has = trait.witness @Has_i32 for @Has[i32]
  %m = trait.witness @Marker_i64 for @Marker[i64]
  %ev = trait.method.call %has @Has[i32]::@requirement_0(%m) : (!trait.claim<@Marker[i64] by @Marker_i64>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [i64]>]> by @Has_i32
  %r = trait.method.call %ev @Marker[!trait.proj<@Has[i32], "A", [i64]>]::@mark(%x) : (i64) -> i64
  return %r : i64
}
