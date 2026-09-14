// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// The premise the witness leaves standing is read by the stage through what
// impl selection settled, so a citation of an impl that does not apply is
// refused there. Here only @Has_o applies at i32, Has[i32]::Out is i8, and @I's
// premise is false.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Other[!trait.poly<0>] {}
trait.impl private @Marker_i16 for @Marker[i16] {}
trait.impl private @Other_i32 for @Other[i32] {}

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
  // expected-error @below {{impl '@I' applies where '!trait.proj<@Has[i32], "Out">' = 'i64', and after instantiate-monomorphs nothing makes 'i8' and 'i64' one type at '!trait.claim<@T[i32] by @I>'}}
  %w = trait.witness @I for @T[i32]
  %r = trait.method.call %w @T[i32]::@m() : () -> i64 by @I
  return %r : i64
}

// -----

// The same premise where neither impl of @Has applies at i32, so selection
// settles the projection to nothing and the premise is decided nowhere. A
// premise nothing settles is not one the citation may stand on.

trait.trait private @Marker[!trait.poly<0>] {}
trait.trait private @Other[!trait.poly<0>] {}
trait.impl private @Marker_i16 for @Marker[i16] {}
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
  // expected-error @below {{impl '@I' applies where '!trait.proj<@Has[i32], "Out">' = 'i64', and after instantiate-monomorphs nothing makes '!trait.proj<@Has[i32], "Out">' and 'i64' one type at '!trait.claim<@T[i32] by @I>'}}
  %w = trait.witness @I for @T[i32]
  %r = trait.method.call %w @T[i32]::@m() : () -> i64 by @I
  return %r : i64
}
