// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A quantified requirement, `forall X where Marker[X] -> Marker[Has[S]::A<X>]`,
// is a bodiless evidence member of the trait: its parameters after the trait's
// own application are the premise claims and its result the conclusion. An
// impl provides it with a member function whose body returns the evidence, and
// a caller selects it with a member call whose result is a claim;
// monomorphization lowers the call through that claim to the impl the evidence
// names.

// ROUNDTRIP:      trait.trait private @Has(!trait.claim<@Has[!trait.poly<0>]>) assoc {A = 1} members [@Has.requirement_0]
// ROUNDTRIP-NEXT: func.func private @Has.requirement_0(!trait.claim<@Has[!trait.poly<0>]>, !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
// ROUNDTRIP:      trait.func.call @Has.requirement_0(%{{.*}}, %{{.*}}) : (!trait.claim<@Has[i32] by @Has_i32>, !trait.claim<@Marker[i64] by @Marker_i64>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [i64]>]>

// CHECK: {{^}}42{{$}}

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(!trait.claim<@Marker[!S]>) members [@Marker.mark]
func.func private @Marker.mark(!trait.claim<@Marker[!S]>, i64) -> i64
trait.trait private @Has(!trait.claim<@Has[!S]>) assoc {A = 1} members [@Has.requirement_0]
func.func private @Has.requirement_0(!trait.claim<@Has[!S]>, !trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) members [@Marker.mark -> @Marker_i64.mark] {
}
func.func private @Marker_i64.mark(%self: !trait.claim<@Marker[i64]>, %x: i64) -> i64 {
  %c = arith.constant 1 : i64
  %r = arith.addi %x, %c : i64
  return %r : i64
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) bindings {A = i64} members [@Has.requirement_0 -> @Has_i32.requirement_0] {
}
func.func private @Has_i32.requirement_0(%self: !trait.claim<@Has[i32]>, %m: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[i64] by @Marker_i64> {
  %r = trait.witness @Marker_i64 for @Marker[i64]
  return %r : !trait.claim<@Marker[i64] by @Marker_i64>
}
func.func @main() -> i64 {
  %x = arith.constant 41 : i64
  %has = trait.witness @Has_i32 for @Has[i32]
  %m = trait.witness @Marker_i64 for @Marker[i64]
  %ev = trait.func.call @Has.requirement_0(%has, %m) : (!trait.claim<@Has[i32] by @Has_i32>, !trait.claim<@Marker[i64] by @Marker_i64>) -> !trait.claim<@Marker[!trait.proj<@Has[i32], "A", [i64]>]>
  %r = trait.func.call @Marker.mark(%ev, %x) : (!trait.claim<@Marker[!trait.proj<@Has[i32], "A", [i64]>]>, i64) -> i64
  return %r : i64
}
