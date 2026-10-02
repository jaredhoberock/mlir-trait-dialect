// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A proof over a projection whose two impls bind each other's associated type
// spells an associated-type binding cycle: @Loop[i32]'s Output is @Loop[i64]'s
// Output and back. The projection has no normal form. The proof's derive cites
// an impl whose header spells the projection identically, so declaring it
// normalizes nothing; the instance that uses it resolves the projection
// through the module's impls, and the nonconverging resolution is reported as
// a clean diagnostic at the use -- it neither aborts the process nor runs the
// cyclic proof.

// CHECK: error: 'trait.witness' op projection normalization did not converge within 64 iterations for type '!trait.claim<@Wants[!trait.proj<@Loop[i32], "Output">]>'

!T = !trait.poly<0>

trait.trait private @Loop(%self: !trait.claim<@Loop[!T]>) {
  trait.assoc_type @Output
}

trait.impl private @Loop_i32(%self: !trait.claim<@Loop[i32]>) {
  trait.assoc_type @Output = !trait.proj<@Loop[i64], "Output">
}

trait.impl private @Loop_i64(%self: !trait.claim<@Loop[i64]>) {
  trait.assoc_type @Output = !trait.proj<@Loop[i32], "Output">
}

!W = !trait.poly<1>
trait.trait private @Wants(%self: !trait.claim<@Wants[!W]>) { trait.method @m() -> i64 }

trait.impl private @Wants_impl(%self: !trait.claim<@Wants[!trait.proj<@Loop[i32], "Output">]>) {
  trait.method @m() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

trait.proof private @p {
  %d = trait.derive @Wants[!trait.proj<@Loop[i32], "Output">] from @Wants_impl given()
  trait.return %d : !trait.claim<@Wants[!trait.proj<@Loop[i32], "Output">]>
}

func.func @main() -> i64 {
  %w = trait.witness @p for @Wants[!trait.proj<@Loop[i32], "Output">]
  %v = trait.method.call %w @Wants[!trait.proj<@Loop[i32], "Output">]::@m() : () -> i64 by @p
  return %v : i64
}
