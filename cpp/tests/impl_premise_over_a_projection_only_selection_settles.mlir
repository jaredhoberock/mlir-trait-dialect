// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @I applies where Has[i32]::Out is i64. Two impls of @Has bind @Has[i32], and
// the impls alone say nothing about which serves it: @Has_m applies where
// Marker[i32] holds and @Has_o where Other[i32] does, and reading a where
// clause is impl selection's work, not a verifier's. So the witness leaves the
// premise standing, and the stage decides it through what selection settled:
// only @Has_m applies at i32, Has[i32]::Out is i64, and @I applies.

// CHECK-LABEL: func.func @main
// CHECK: call @I_m

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Other[!trait.poly<0>] {}
trait.impl private @Marker_i32 for @Marker[i32] {}
trait.impl private @Other_i8 for @Other[i8] {}

trait.trait private @Has[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Has_m for @Has[!trait.poly<0>] where [@Marker[!trait.poly<0>]] { trait.assoc_type @Out = i64 }
trait.impl private @Has_o for @Has[!trait.poly<0>] where [@Other[!trait.poly<0>]] { trait.assoc_type @Out = i8 }

trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
trait.impl private @I for @T[i32] where [!trait.proj<@Has[i32], "Out"> = i64] {
  func.func @m() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}

func.func @main() -> i64 {
  %w = trait.witness @I for @T[i32]
  %r = trait.method.call %w @T[i32]::@m() : () -> i64 by @I
  return %r : i64
}
