// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics | FileCheck %s

// A projection of a trait requirement is the trait's requirement at its
// source's arguments, which names no proof: its evidence is what the source's
// impl returns there, which the stage puts in its place. A call reading the
// projection runs the impl that evidence rests on. A result naming a proof is
// refused where it stands, whether the proof proves another application, is
// another impl of the same one, or is the impl the evidence rests on.

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) { trait.method @m() -> i64 }
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {
  trait.method @m() -> i64 { %c = arith.constant 8 : i64  trait.return %c : i64 }
}
trait.trait private @Has(%self: !trait.claim<@Has[!T]>) -> !trait.claim<@T[!T]> {}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  %w = trait.witness @T_i32 for @T[i32]
  trait.return %w : !trait.claim<@T[i32] by @T_i32>
}
func.func @main() -> i64 {
  %h = trait.witness @Has_i32 for @Has[i32]
  // expected-error @below {{type mismatch: expected '!trait.claim<@T[i32]>' but found '!trait.claim<@T[i32] by @T_i64>'}}
  %c = trait.project %h[0] : !trait.claim<@Has[i32] by @Has_i32> -> !trait.claim<@T[i32] by @T_i64>
  %r = trait.method.call %c @m() : !trait.claim<@T[i32] by @T_i64>, () -> i64
  return %r : i64
}

// -----

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
  // expected-error @below {{type mismatch: expected '!trait.claim<@Mark[i32]>' but found '!trait.claim<@Mark[i32] by @Seven>'}}
  %m = trait.project %a[0] : !trait.claim<@A[i32] by @I> -> !trait.claim<@Mark[i32] by @Seven>
  %v = trait.method.call %m @value() : !trait.claim<@Mark[i32] by @Seven>, () -> i64
  return %v : i64
}

// -----

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
  // expected-error @below {{type mismatch: expected '!trait.claim<@Mark[i32]>' but found '!trait.claim<@Mark[i32] by @Nine>'}}
  %m = trait.project %a[0] : !trait.claim<@A[i32] by @I> -> !trait.claim<@Mark[i32] by @Nine>
  %v = trait.method.call %m @value() : !trait.claim<@Mark[i32] by @Nine>, () -> i64
  return %v : i64
}

// -----

// The same projection naming no proof: @I returns a derive of @Mark[i32] from
// @Nine, so the call runs @Nine's method.

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
  %m = trait.project %a[0] : !trait.claim<@A[i32] by @I> -> !trait.claim<@Mark[i32]>
  %v = trait.method.call %m @value() : !trait.claim<@Mark[i32]>, () -> i64
  return %v : i64
}
