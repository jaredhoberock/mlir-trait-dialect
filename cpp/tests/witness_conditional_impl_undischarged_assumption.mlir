// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// A conditional impl cited by a witness proj_resolve that supplies no premise
// for the impl's assumption. @Has_tuple binds @Has[tuple<!U>]::Out to i64 and
// requires @X[!U]; for !U = i32 the module carries no impl of @X[i32], and the
// witness carries no premise supplying it. The witness's binding is correct,
// but obligation-aware verification additionally demands the cited impl's own
// assumptions be discharged by the witness's premises, so the witness is refused.
// Supplying the @X[i32] premise is what discharges the assumption (see
// witness_equality_obligation_discharge).

!U = !trait.poly<0>

trait.trait @X[!U] {}

trait.trait @Has[!U] {
  trait.assoc_type @Out
}

trait.impl @Has_tuple for @Has[tuple<!U>] where [@X[!U]] {
  trait.assoc_type @Out = i64
}

// CHECK: cited impl '@Has_tuple' has an undischarged assumption '!trait.claim<@X[i32]>'
func.func @f(%v: !trait.proj<@Has[tuple<i32>], "Out">) -> i64 {
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
