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
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped[!T] where [@Mark[!T]] {
  trait.method @value() -> i64 {
    %p = trait.assume 0 : !trait.claim<@Mark[!T]>
    %v = trait.method.call %p @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @W for @Wrapped[i32] {}
trait.proof private @W9 proves @W[] for @Wrapped[i32] given [@Nine]
trait.proof private @O9 proves @O[] for @Outer[i32] given [@W9]
trait.trait private @Outer[!T] where [@Wrapped[!T]] {}
trait.impl private @O for @Outer[i32] {}
func.func private @f(%x: !T) -> i64 {
  %a = trait.allege @Outer[!T]
  %w = trait.project %a[0] : !trait.claim<@Outer[!T]> -> !trait.claim<@Wrapped[!T]>
  %v = trait.method.call %w @Wrapped[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %first = trait.allege @Wrapped[i32]
  %v0 = trait.method.call %first @Wrapped[i32]::@value() : () -> i64
  %x = arith.constant 0 : i32
  %v = trait.func.call @f(%x) : (i32) -> i64
  return %v : i64
}
