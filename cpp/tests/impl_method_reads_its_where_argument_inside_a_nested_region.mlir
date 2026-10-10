// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=INSTANCE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The prelude's Into impl: its where entry From[U, T] is the impl's block
// argument, and its method reads it inside a nested region. The instance cut
// for Into[i32, i64] replaces the argument by the evidence the proof its call
// resolved to computed for it, a witness of @From_i64_i32.

// ROUNDTRIP:      trait.impl private @Into_impl(%self: !trait.claim<@Into[!trait.poly<0>, !trait.poly<1>]>, %from: !trait.claim<@From[!trait.poly<1>, !trait.poly<0>]>) {
// ROUNDTRIP:        scf.execute_region
// ROUNDTRIP-NEXT:     trait.method.call %from @From[!trait.poly<1>, !trait.poly<0>]::@from(

// INSTANCE:      func.func private @Into_impl_{{.*}}_into(%{{.*}}: !trait.claim<@Into[i32, i64] by @{{.*}}>, %[[X:.*]]: i32) -> i64 {
// INSTANCE-NEXT:   %[[FROM:.*]] = trait.witness @From_i64_i32 for @From[i64, i32]
// INSTANCE:        call @From_i64_i32_{{.*}}_from(%[[FROM]], %[[X]])

// CHECK: {{^}}8{{$}}

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @From(%self: !trait.claim<@From[!trait.poly<0>, !trait.poly<1>]>) {
  trait.method @from(!trait.poly<1>) -> !trait.poly<0>
}
trait.trait private @Into(%self: !trait.claim<@Into[!T, !U]>) {
  trait.method @into(!T) -> !U
}
trait.impl private @Into_impl(%self: !trait.claim<@Into[!T, !U]>, %from: !trait.claim<@From[!U, !T]>) {
  trait.method @into(%x: !T) -> !U {
    %r = scf.execute_region -> !U {
      %v = trait.method.call %from @From[!U, !T]::@from(%x) : (!T) -> !U
      scf.yield %v : !U
    }
    trait.return %r : !U
  }
}
trait.impl private @From_i64_i32(%self: !trait.claim<@From[i64, i32]>) {
  trait.method @from(%x: i32) -> i64 {
    %c = arith.constant 1 : i64
    %w = arith.extsi %x : i32 to i64
    %r = arith.addi %w, %c : i64
    trait.return %r : i64
  }
}
func.func @main() -> i64 {
  %x = arith.constant 7 : i32
  %c = trait.allege @Into[i32, i64]
  %r = trait.method.call %c @Into[i32, i64]::@into(%x) : (i32) -> i64
  return %r : i64
}
