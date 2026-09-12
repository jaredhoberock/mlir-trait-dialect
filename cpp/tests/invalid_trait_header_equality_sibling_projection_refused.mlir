// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A trait's header requires a sibling-projection equality (@Sib[Self]::Elem =
// f32). The requirement is an obligation the impl owes, so the two endpoints
// must be the same type: this impl declares no witness, nothing here reduces
// @Sib[i64]::Elem, and a projection nothing reduces is equal to itself alone.
// The impl is refused. The impl that satisfies such a requirement declares the
// witness reducing the endpoint; impl_satisfies_trait_equality_requirement.mlir
// is that row.

!S = !trait.poly<0>

trait.trait private @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64 for @Sib[i64] {
  trait.assoc_type @Elem = i64
}

trait.trait private @T[!S] where [!trait.proj<@Sib[!S], "Elem"> = f32] {
  func.func private @id(!S) -> !S
}

// expected-error @below {{does not satisfy trait-header equality requirement '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>': '!trait.proj<@Sib[i64], "Elem">' and 'f32' are not the same type}}
trait.impl private @T_i64 for @T[i64] {
  func.func @id(%x: i64) -> i64 {
    return %x : i64
  }
}
