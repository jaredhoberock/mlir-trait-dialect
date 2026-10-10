// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @V_blanket applies where @Has[T]::Out is i64. Two conditional impls could
// bind @Has[i8], and only @Has_m holds there; @pv supplies the equality entry
// with a proj_resolve witness citing @Has_m over the proof @ma of its premise.
// The proof's premises are decided in its own body. Split 1 carries the proof
// to an instance at a witness; split 2 takes the proven claim through a
// function parameter.

trait.trait private @Marker(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.trait private @Other(%self: !trait.claim<@Other[!trait.poly<0>]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Marker_any(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.impl private @Other_i32(%self: !trait.claim<@Other[i32]>) {}
trait.impl private @Has_m(%self: !trait.claim<@Has[!trait.poly<0>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) { trait.assoc_type @Out = i64 }
trait.impl private @Has_o(%self: !trait.claim<@Has[!trait.poly<0>]>, %other: !trait.claim<@Other[!trait.poly<0>]>) { trait.assoc_type @Out = i32 }
trait.trait private @V(%self: !trait.claim<@V[!trait.poly<0>]>) { trait.method @v() -> i64 }
trait.impl private @V_blanket(%self: !trait.claim<@V[!trait.poly<0>]>, %has: !trait.claim<@Has[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Has[!trait.poly<0>], "Out"> = i64>) {
  trait.method @v() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.proof private @ma {
  %d = trait.derive @Marker[i8] from @Marker_any given()
  trait.return %d : !trait.claim<@Marker[i8]>
}
trait.proof private @hm {
  %p0 = trait.witness @ma for @Marker[i8]
  %d = trait.derive @Has[i8] from @Has_m given(%p0) : (!trait.claim<@Marker[i8] by @ma>)
  trait.return %d : !trait.claim<@Has[i8]>
}
trait.proof private @pv {
  %p0 = trait.witness @hm for @Has[i8]
  %m = trait.witness @ma for @Marker[i8]
  %p1 = trait.witness proj_resolve !trait.proj<@Has[i8], "Out"> resolves i64 by @Has_m
    given(%m) : (!trait.claim<@Marker[i8] by @ma>)
    : !trait.claim<!trait.proj<@Has[i8], "Out"> = i64>
  %d = trait.derive @V[i8] from @V_blanket given(%p0, %p1) : (!trait.claim<@Has[i8] by @hm>, !trait.claim<!trait.proj<@Has[i8], "Out"> = i64>)
  trait.return %d : !trait.claim<@V[i8]>
}

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

trait.trait private @Marker(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.trait private @Other(%self: !trait.claim<@Other[!trait.poly<0>]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Marker_any(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.impl private @Other_i32(%self: !trait.claim<@Other[i32]>) {}
trait.impl private @Has_m(%self: !trait.claim<@Has[!trait.poly<0>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) { trait.assoc_type @Out = i64 }
trait.impl private @Has_o(%self: !trait.claim<@Has[!trait.poly<0>]>, %other: !trait.claim<@Other[!trait.poly<0>]>) { trait.assoc_type @Out = i32 }
trait.trait private @V(%self: !trait.claim<@V[!trait.poly<0>]>) { trait.method @v() -> i64 }
trait.impl private @V_blanket(%self: !trait.claim<@V[!trait.poly<0>]>, %has: !trait.claim<@Has[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Has[!trait.poly<0>], "Out"> = i64>) {
  trait.method @v() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.proof private @ma {
  %d = trait.derive @Marker[i8] from @Marker_any given()
  trait.return %d : !trait.claim<@Marker[i8]>
}
trait.proof private @hm {
  %p0 = trait.witness @ma for @Marker[i8]
  %d = trait.derive @Has[i8] from @Has_m given(%p0) : (!trait.claim<@Marker[i8] by @ma>)
  trait.return %d : !trait.claim<@Has[i8]>
}
trait.proof private @pv {
  %p0 = trait.witness @hm for @Has[i8]
  %m = trait.witness @ma for @Marker[i8]
  %p1 = trait.witness proj_resolve !trait.proj<@Has[i8], "Out"> resolves i64 by @Has_m
    given(%m) : (!trait.claim<@Marker[i8] by @ma>)
    : !trait.claim<!trait.proj<@Has[i8], "Out"> = i64>
  %d = trait.derive @V[i8] from @V_blanket given(%p0, %p1) : (!trait.claim<@Has[i8] by @hm>, !trait.claim<!trait.proj<@Has[i8], "Out"> = i64>)
  trait.return %d : !trait.claim<@V[i8]>
}

// CHECK-NOT: trait.
// CHECK: func.func private @[[V2:V_blanket_[a-z0-9]+]]_v() -> i64
// CHECK: func.func @f() -> i64
// CHECK: call @[[V2]]_v() : () -> i64
// CHECK-NOT: trait.
func.func @f(%c: !trait.claim<@V[i8] by @pv>) -> i64 {
  %r = trait.method.call %c @V[i8]::@v() : () -> i64 by @pv
  return %r : i64
}
