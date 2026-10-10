// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=SELECT
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @I applies where Has[i32]::Out is i64. Two impls of @Has bind @Has[i32], and
// the impls alone say nothing about which serves it: @Has_m applies where
// Marker[i32] holds and @Has_o where Other[i32] does, and reading a where
// clause is impl selection's work, not a verifier's. @main alleges @T[i32], and
// the stage decides the premise through what selection settles: only @Has_m
// applies at i32, so the proof of @I cites @Has_m's binding for
// Has[i32]::Out = i64.

// SELECT: trait.proof private @I_p {
// SELECT: trait.witness proj_resolve !trait.proj<@Has[i32], "Out"> resolves i64 by @Has_m[i32] given
// SELECT: trait.derive @T[i32] from @I given

// CHECK-LABEL: func.func @main
// CHECK: call @I_{{h[0-9a-f]+}}_m

trait.trait private @Marker(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.trait private @Other(%self: !trait.claim<@Other[!trait.poly<0>]>) {}
trait.impl private @Marker_i32(%self: !trait.claim<@Marker[i32]>) {}
trait.impl private @Other_i8(%self: !trait.claim<@Other[i8]>) {}

trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Has_m(%self: !trait.claim<@Has[!trait.poly<0>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) { trait.assoc_type @Out = i64 }
trait.impl private @Has_o(%self: !trait.claim<@Has[!trait.poly<0>]>, %other: !trait.claim<@Other[!trait.poly<0>]>) { trait.assoc_type @Out = i8 }

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @I(%self: !trait.claim<@T[i32]>, %out: !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>) {
  trait.method @m() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

func.func @main() -> i64 {
  %w = trait.allege @T[i32]
  %r = trait.method.call %w @T[i32]::@m() : () -> i64
  return %r : i64
}
