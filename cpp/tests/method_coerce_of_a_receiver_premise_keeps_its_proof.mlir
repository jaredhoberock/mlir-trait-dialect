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
trait.trait private @Assoc(%self: !trait.claim<@Assoc[!T]>) { trait.assoc_type @A }
trait.impl private @Assoc_i32(%self: !trait.claim<@Assoc[i32]>) { trait.assoc_type @A = i32 }
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) { trait.method @run() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>, %mark: !trait.claim<@Mark[i32]>) {
  trait.method @run() -> i64 {
    %e = trait.witness proj_resolve !trait.proj<@Assoc[i32], "A"> resolves i32 by @Assoc_i32 : !trait.claim<!trait.proj<@Assoc[i32], "A"> = i32>
    %m = trait.coerce %mark : !trait.claim<@Mark[i32]> to !trait.claim<@Mark[!trait.proj<@Assoc[i32], "A">]> via (%e) : (!trait.claim<!trait.proj<@Assoc[i32], "A"> = i32>)
    %v = trait.method.call %m @Mark[!trait.proj<@Assoc[i32], "A">]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H {
  %p0 = trait.witness @Seven for @Mark[i32]
  %d = trait.derive @Host[i32] from @Host_i32 given(%p0) : (!trait.claim<@Mark[i32] by @Seven>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %v = trait.method.call %h @Host[i32]::@run() : () -> i64 by @H
  return %v : i64
}
