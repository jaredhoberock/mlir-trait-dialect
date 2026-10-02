// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @main alleges @Wrapped[i32] first, which selection proves by @WA, the
// first of its two proofs, over @MarkA. @f then alleges @Outer[i32], which
// selection proves by @OB over @WB, and projects @Wrapped[i32] off it. The
// projection's result is spelled unproven, so selection offers its own proof
// of @Wrapped[i32], @WA; the source supplies @WB at the index, a proof over
// the other impl of @Mark[i32]. Selection's proof is confirmed against the
// source before it is witnessed, and the projection is refused rather than
// run through @MarkA.

// CHECK: error: 'trait.project' op names @WA, which its source does not supply at index 0: the evidence there is @WB

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @MarkA for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @MarkB for @Mark[i32] {
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
trait.proof private @WA proves @W[] for @Wrapped[i32] given [@MarkA]
trait.proof private @WB proves @W[] for @Wrapped[i32] given [@MarkB]
trait.proof private @OB proves @O[] for @Outer[i32] given [@WB]
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
