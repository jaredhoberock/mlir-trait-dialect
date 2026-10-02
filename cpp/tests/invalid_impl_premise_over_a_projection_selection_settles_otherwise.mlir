// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A citation of @I supplies its equality premise, here an allegation, which
// the stage proves through what impl selection settles, so a citation of an
// impl that does not apply is refused where the premise is alleged. Here only
// @Has_o applies at i32, Has[i32]::Out is i8, and @I's premise is false.

trait.trait private @Marker(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.trait private @Other(%self: !trait.claim<@Other[!trait.poly<0>]>) {}
trait.impl private @Marker_i16(%self: !trait.claim<@Marker[i16]>) {}
trait.impl private @Other_i32(%self: !trait.claim<@Other[i32]>) {}

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
  // expected-error @below {{alleges '!trait.proj<@Has[i32], "Out">' = 'i64', and impl selection resolves its sides to 'i8' and 'i64'}}
  // expected-error @below {{unresolved monomorphic trait.allege after resolve-impls}}
  %eq = trait.allege !trait.proj<@Has[i32], "Out"> = i64
  %w = trait.derive @T[i32] from @I given(%eq) : (!trait.claim<!trait.proj<@Has[i32], "Out"> = i64>)
  %r = trait.method.call %w @T[i32]::@m() : () -> i64
  return %r : i64
}

// -----

// The same premise where neither impl of @Has applies at i32, so selection
// settles the projection to nothing and the alleged premise is refused: a
// premise nothing settles is not one the citation may stand on.

trait.trait private @Marker(%self: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.trait private @Other(%self: !trait.claim<@Other[!trait.poly<0>]>) {}
trait.impl private @Marker_i16(%self: !trait.claim<@Marker[i16]>) {}
trait.impl private @Other_i8(%self: !trait.claim<@Other[i8]>) {}

trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @Has_m(%self: !trait.claim<@Has[!trait.poly<0>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) { trait.assoc_type @Out = i64 }
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @Has_o(%self: !trait.claim<@Has[!trait.poly<0>]>, %other: !trait.claim<@Other[!trait.poly<0>]>) { trait.assoc_type @Out = i8 }

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @I(%self: !trait.claim<@T[i32]>, %out: !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>) {
  trait.method @m() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

func.func @main() -> i64 {
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.claim<@Has[i32]>'}}
  // expected-error @below {{unresolved monomorphic trait.allege after resolve-impls}}
  %eq = trait.allege !trait.proj<@Has[i32], "Out"> = i64
  %w = trait.derive @T[i32] from @I given(%eq) : (!trait.claim<!trait.proj<@Has[i32], "Out"> = i64>)
  %r = trait.method.call %w @T[i32]::@m() : () -> i64
  return %r : i64
}
