// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @V_blanket applies where @Has[T]::Out is i64. Two conditional impls could
// bind @Has[i8], and only @Has_m holds there, so the premise reads through the
// subproof @pv cites for its @Has obligation. That subproof's claim is the
// obligation at the application the citation names, @Has[i8], which is the key
// the premise's projection is looked up by. Split 1 names the instance at a
// witness; split 2 takes the proven claim through a function parameter, so the
// premise is read on the method call's derivation path instead.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Other[!trait.poly<0>] {}
trait.trait private @Has[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Marker_any for @Marker[!trait.poly<0>] {}
trait.impl private @Other_i32 for @Other[i32] {}
trait.impl private @Has_m for @Has[!trait.poly<0>] where [@Marker[!trait.poly<0>]] { trait.assoc_type @Out = i64 }
trait.impl private @Has_o for @Has[!trait.poly<0>] where [@Other[!trait.poly<0>]] { trait.assoc_type @Out = i32 }
trait.trait private @V[!trait.poly<0>] { func.func private @v() -> i64 }
trait.impl private @V_blanket for @V[!trait.poly<0>] where [@Has[!trait.poly<0>], !trait.proj<@Has[!trait.poly<0>], "Out"> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.proof private @ma proves @Marker_any for @Marker[!trait.poly<0>] given []
trait.proof private @hm proves @Has_m for @Has[!trait.poly<0>] given [@ma]
trait.proof private @pv proves @V_blanket for @V[!trait.poly<0>] given [@hm]

// CHECK-NOT: trait.
// CHECK: func.func private @[[V:V_blanket_[a-z0-9]+]]_v() -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @[[V]]_v() : () -> i64
// CHECK-NOT: trait.
func.func @main() -> i64 {
  %w = trait.witness @pv for @V[i8]
  %r = trait.method.call %w @V[i8]::@v() : () -> i64 by @pv
  return %r : i64
}

// -----

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Other[!trait.poly<0>] {}
trait.trait private @Has[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Marker_any for @Marker[!trait.poly<0>] {}
trait.impl private @Other_i32 for @Other[i32] {}
trait.impl private @Has_m for @Has[!trait.poly<0>] where [@Marker[!trait.poly<0>]] { trait.assoc_type @Out = i64 }
trait.impl private @Has_o for @Has[!trait.poly<0>] where [@Other[!trait.poly<0>]] { trait.assoc_type @Out = i32 }
trait.trait private @V[!trait.poly<0>] { func.func private @v() -> i64 }
trait.impl private @V_blanket for @V[!trait.poly<0>] where [@Has[!trait.poly<0>], !trait.proj<@Has[!trait.poly<0>], "Out"> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.proof private @ma proves @Marker_any for @Marker[!trait.poly<0>] given []
trait.proof private @hm proves @Has_m for @Has[!trait.poly<0>] given [@ma]
trait.proof private @pv proves @V_blanket for @V[!trait.poly<0>] given [@hm]

// CHECK-NOT: trait.
// CHECK: func.func private @[[V2:V_blanket_[a-z0-9]+]]_v() -> i64
// CHECK: func.func @f() -> i64
// CHECK: call @[[V2]]_v() : () -> i64
// CHECK-NOT: trait.
func.func @f(%c: !trait.claim<@V[i8] by @pv>) -> i64 {
  %r = trait.method.call %c @V[i8]::@v() : () -> i64 by @pv
  return %r : i64
}
