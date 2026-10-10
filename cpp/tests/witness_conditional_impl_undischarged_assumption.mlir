// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// A conditional impl cited by a witness proj_resolve that supplies no premise
// for the impl's where entry. @Has_tuple binds @Has[tuple<!U>]::Out to i64 and
// takes @X[!U]; the witness supplies no claim at that position. The witness's
// binding is correct, but a citation supplies one claim per where entry of the
// impl it cites, so the witness is refused. Supplying the @X[i32] premise is
// what discharges the entry (see witness_equality_obligation_discharge).

!U = !trait.poly<0>

trait.trait private @X(%self: !trait.claim<@X[!U]>) {}

trait.trait private @Has(%self: !trait.claim<@Has[!U]>) {
  trait.assoc_type @Out
}

trait.impl private @Has_tuple(%self: !trait.claim<@Has[tuple<!U>]>, %x: !trait.claim<@X[!U]>) {
  trait.assoc_type @Out = i64
}

// CHECK: error: 'trait.witness' op impl '@Has_tuple' states 1 premises, and the witness supplies 0
func.func @f(%v: !trait.proj<@Has[tuple<i32>], "Out">) -> i64 {
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple[i32]
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
