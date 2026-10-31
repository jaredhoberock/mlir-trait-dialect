// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=INSTANCE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The prelude's Into impl: its where entry From[U, T] is the impl's block
// argument and its member function's leading parameter after the self claim,
// which the function reads inside a nested region. The instance a call of the
// member at Into[i32, i64] reaches replaces that parameter by the evidence the
// proof the call resolved to computed for it, a witness of @From_i64_i32, and
// takes the receiver and the member's input alone.

// ROUNDTRIP:      trait.impl private @Into_impl(%self: !trait.claim<@Into[!trait.poly<0>, !trait.poly<1>]>, %from: !trait.claim<@From[!trait.poly<1>, !trait.poly<0>]>) members [@Into.into -> @Into_impl.into]
// ROUNDTRIP:      func.func private @Into_impl.into(%{{[^:]*}}: !trait.claim<@Into[!trait.poly<0>, !trait.poly<1>]>, %[[FROM:[^:]*]]: !trait.claim<@From[!trait.poly<1>, !trait.poly<0>]>, %[[X:[^:]*]]: !trait.poly<0>)
// ROUNDTRIP:        scf.execute_region
// ROUNDTRIP-NEXT:     trait.func.call @From.from(%[[FROM]], %[[X]])

// INSTANCE:      func.func private @[[INTO:Into_impl.into_h[0-9a-f]+]](%{{[^:]*}}: !trait.claim<@Into[i32, i64] by @{{.*}}>, %[[X:[^:]*]]: i32) -> i64 {
// INSTANCE-NEXT:   %[[FROM:.*]] = trait.witness @From_i64_i32 for @From[i64, i32]
// INSTANCE:        call @From_i64_i32.from_{{.*}}(%[[FROM]], %[[X]])
// INSTANCE:      func.func @main
// INSTANCE:        call @[[INTO]](%{{[^,]*}}, %{{[^,]*}})

// CHECK: {{^}}8{{$}}

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @From(!trait.claim<@From[!trait.poly<0>, !trait.poly<1>]>) members [@From.from]
func.func private @From.from(!trait.claim<@From[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<1>) -> !trait.poly<0>
trait.trait private @Into(!trait.claim<@Into[!T, !U]>) members [@Into.into]
func.func private @Into.into(!trait.claim<@Into[!T, !U]>, !T) -> !U
trait.impl private @Into_impl(%self: !trait.claim<@Into[!T, !U]>, %from: !trait.claim<@From[!U, !T]>) members [@Into.into -> @Into_impl.into] {
}
func.func private @Into_impl.into(%self: !trait.claim<@Into[!T, !U]>, %from: !trait.claim<@From[!U, !T]>, %x: !T) -> !U {
  %r = scf.execute_region -> !U {
    %v = trait.func.call @From.from(%from, %x) : (!trait.claim<@From[!U, !T]>, !T) -> !U
    scf.yield %v : !U
  }
  return %r : !U
}
trait.impl private @From_i64_i32(%self: !trait.claim<@From[i64, i32]>) members [@From.from -> @From_i64_i32.from] {
}
func.func private @From_i64_i32.from(%self: !trait.claim<@From[i64, i32]>, %x: i32) -> i64 {
  %c = arith.constant 1 : i64
  %w = arith.extsi %x : i32 to i64
  %r = arith.addi %w, %c : i64
  return %r : i64
}
func.func @main() -> i64 {
  %x = arith.constant 7 : i32
  %c = trait.allege @Into[i32, i64]
  %r = trait.func.call @Into.into(%c, %x) : (!trait.claim<@Into[i32, i64]>, i32) -> i64
  return %r : i64
}
