// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run reads its impl's @Mark[i32] entry, which the receiver's proof @H
// discharges with @Seven (7), and coerces it to @Mark[@Assoc[i32]::A] through
// @Assoc_i32's binding of A to i32. The equality respells the application the
// claim proves and never its proof, so the coerced claim carries @Seven and
// the call through it runs @Seven's method although @Nine also implements
// @Mark[i32].

// CHECK: {{^}}7{{$}}

!T = !trait.poly<0>
trait.trait private @Assoc[!T] { trait.assoc_type @A }
trait.impl private @Assoc_i32 for @Assoc[i32] { trait.assoc_type @A = i32 }
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.trait private @Host[!T] { trait.method @run() -> i64 }
trait.impl private @Seven for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Host_i32 for @Host[i32] where [@Mark[i32]] {
  trait.method @run() -> i64 {
    %s = trait.assume 0 : !trait.claim<@Mark[i32]>
    %e = trait.witness proj_resolve !trait.proj<@Assoc[i32], "A"> resolves i32 by @Assoc_i32 : !trait.claim<!trait.proj<@Assoc[i32], "A"> = i32>
    %m = trait.coerce %s : !trait.claim<@Mark[i32]> to !trait.claim<@Mark[!trait.proj<@Assoc[i32], "A">]> via (%e) : (!trait.claim<!trait.proj<@Assoc[i32], "A"> = i32>)
    %v = trait.method.call %m @Mark[!trait.proj<@Assoc[i32], "A">]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %v = trait.method.call %h @Host[i32]::@run() : () -> i64 by @H
  return %v : i64
}
