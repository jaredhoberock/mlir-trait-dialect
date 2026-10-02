// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped requires @Mark of its associated type at every argument through its
// evidence method @requirement_0, which @W implements by returning @Nine's
// witness; @run calls the evidence method off its parameter at i1, while the
// receiver's proof @PH discharges the impl's own @Mark[i32] entry with @Seven.
// The call is replaced by @W's body, so it runs @Nine's method, 9, whatever
// the receiver supplies for the same claim.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
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
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Wrapped[!T], "Item", [!U]>]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  trait.assoc_type @Item<[!trait.poly<1>]> = i32
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [!U]>] by @Nine> {
    %n = trait.witness @Nine for @Mark[i32]
    %e = trait.witness proj_resolve !trait.proj<@Wrapped[i32], "Item", [!U]> resolves i32 by @W : !trait.claim<!trait.proj<@Wrapped[i32], "Item", [!U]> = i32>
    %m = trait.coerce %n : !trait.claim<@Mark[i32] by @Nine> to !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [!U]>] by @Nine> via (%e) : (!trait.claim<!trait.proj<@Wrapped[i32], "Item", [!U]> = i32>)
    trait.return %m : !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [!U]>] by @Nine>
  }
}
trait.proof private @PW {
  %d = trait.derive @Wrapped[i32] from @W given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) { trait.method @run(!trait.claim<@Wrapped[!T]>) -> i64 }
trait.impl private @H(%self: !trait.claim<@Host[i32]>, %mark: !trait.claim<@Mark[i32]>) {
  trait.method @run(%w: !trait.claim<@Wrapped[i32]>) -> i64 {
    %m = trait.method.call %w @Wrapped[i32]::@requirement_0() : () -> !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>]>
    %v = trait.method.call %m @Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @PH {
  %p0 = trait.witness @Seven for @Mark[i32]
  %d = trait.derive @Host[i32] from @H given(%p0) : (!trait.claim<@Mark[i32] by @Seven>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @PH for @Host[i32]
  %w = trait.witness @PW for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @PW>) -> i64 by @PH
  return %v : i64
}
