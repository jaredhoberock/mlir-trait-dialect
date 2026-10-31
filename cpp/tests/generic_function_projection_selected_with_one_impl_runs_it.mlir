// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @main alleges @Wrapped[i32] first, which selection proves by @W9. @f then
// alleges @Outer[i32], proven by @O9 over @W9, and projects @Wrapped[i32] off
// it; the projection's result is spelled unproven, and selection's proof of
// it, @W9, is the evidence the source supplies at the index, so the instance
// runs @Nine.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Nine.value] {
}
func.func private @Nine.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> members [default @Wrapped.value]
func.func private @Wrapped.value(%self: !trait.claim<@Wrapped[!T]>) -> i64 {
  %p = trait.project %self[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
  %v = trait.func.call @Mark.value(%p) : (!trait.claim<@Mark[!T]>) -> i64
  return %v : i64
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) members [@Wrapped.value -> @Wrapped.value] {
  %req0 = trait.allege @Mark[i32]
  trait.return %req0 : !trait.claim<@Mark[i32]>
}
trait.proof private @W9 {
  %d = trait.derive @Wrapped[i32] from @W given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.proof private @O9 {
  %d = trait.derive @Outer[i32] from @O given()
  trait.return %d : !trait.claim<@Outer[i32]>
}
trait.trait private @Outer(!trait.claim<@Outer[!T]>) -> !trait.claim<@Wrapped[!T]>
trait.impl private @O(%self: !trait.claim<@Outer[i32]>) {
  %req0 = trait.allege @Wrapped[i32]
  trait.return %req0 : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%x: !T) -> i64 {
  %a = trait.allege @Outer[!T]
  %w = trait.project %a[0] : !trait.claim<@Outer[!T]> -> !trait.claim<@Wrapped[!T]>
  %v = trait.func.call @Wrapped.value(%w) : (!trait.claim<@Wrapped[!T]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %first = trait.allege @Wrapped[i32]
  %v0 = trait.func.call @Wrapped.value(%first) : (!trait.claim<@Wrapped[i32]>) -> i64
  %x = arith.constant 0 : i32
  %v = trait.func.call @f(%x) : (i32) -> i64
  return %v : i64
}
