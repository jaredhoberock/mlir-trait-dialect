// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics | FileCheck %s

// @I returns, for @A's requirement, a derive of @Mark[i32] from @Nine. A
// projection of that requirement is replaced by that derive, which names
// @Nine itself once proven, so the projection may spell its result proven only
// by @Nine; spelled by @Seven it names another impl's evidence, and the
// coercion bridging the two refuses to exchange one proof for the other.

!T = !trait.poly<0>
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
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@Mark[!T]> {}
trait.impl private @I(%self: !trait.claim<@A[i32]>) {
  %m = trait.derive @Mark[i32] from @Nine given()
  trait.return %m : !trait.claim<@Mark[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i32]
  // expected-error @below {{may not swap the proof backing claim #trait<application@Mark[i32]>: a coerce compares modulo a proof but may not exchange it for another}}
  %m = trait.project %a[0] : !trait.claim<@A[i32] by @I> -> !trait.claim<@Mark[i32] by @Seven>
  %v = trait.method.call %m @Mark[i32]::@value() : () -> i64 by @Seven
  return %v : i64
}

// -----

// The same projection spelled by @Nine names the derive's own impl, and the
// call runs it.

// CHECK-LABEL: func.func @main() -> i64
// CHECK:         call @Nine_{{h[0-9a-f]+}}_value()

!T = !trait.poly<0>
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
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@Mark[!T]> {}
trait.impl private @I(%self: !trait.claim<@A[i32]>) {
  %m = trait.derive @Mark[i32] from @Nine given()
  trait.return %m : !trait.claim<@Mark[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i32]
  %m = trait.project %a[0] : !trait.claim<@A[i32] by @I> -> !trait.claim<@Mark[i32] by @Nine>
  %v = trait.method.call %m @Mark[i32]::@value() : () -> i64 by @Nine
  return %v : i64
}
