// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A trait's header requires a sibling-projection equality (@Sib[Self]::Elem =
// f32). The requirement is evidence the impl returns, and @Sib_i64 binds
// @Sib[i64]::Elem to i64, so no evidence makes the two endpoints one type. An
// impl alleging the equality is refused where its allegation is decided: the
// instance of a method that projects the requirement off the impl's own claim,
// where the projection is replaced by the allegation, located as the impl
// wrote it.
// The impl that satisfies such a requirement returns the witness reducing the
// endpoint; impl_satisfies_trait_equality_requirement.mlir is that row.

!S = !trait.poly<0>

trait.trait private @Sib(%self: !trait.claim<@Sib[!S]>) {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64(%self: !trait.claim<@Sib[i64]>) {
  trait.assoc_type @Elem = i64
}

trait.trait private @T(%self: !trait.claim<@T[!S]>) -> !trait.claim<!trait.proj<@Sib[!S], "Elem"> = f32> {
  trait.method @get(!S) -> !trait.proj<@Sib[!S], "Elem">
}

trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {
  trait.method @get(%x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
    %c = arith.constant 1.0 : f32
    %e = trait.project %self[0] : !trait.claim<@T[i64]> -> !trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>
    %v = trait.coerce %c : f32 to !trait.proj<@Sib[i64], "Elem"> via (%e) : (!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>)
    trait.return %v : !trait.proj<@Sib[i64], "Elem">
  }
  // expected-error @below {{alleges '!trait.proj<@Sib[i64], "Elem">' = 'f32', and impl selection resolves its sides to 'i64' and 'f32'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>' after instantiate-monomorphs}}
  %alleged = trait.allege !trait.proj<@Sib[i64], "Elem"> = f32
  trait.return %alleged : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>
}

func.func @main(%x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
  %w = trait.witness @T_i64 for @T[i64]
  %r = trait.method.call %w @T[i64]::@get(%x) : (i64) -> !trait.proj<@Sib[i64], "Elem"> by @T_i64
  return %r : !trait.proj<@Sib[i64], "Elem">
}

// -----

// The same requirement with the impl returning the evidence it has: @Sib_i64's
// resolution makes @Sib[i64]::Elem i64, which is not the requirement, so the
// return is refused where it is written.

!S = !trait.poly<0>

trait.trait private @Sib(%self: !trait.claim<@Sib[!S]>) {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64(%self: !trait.claim<@Sib[i64]>) {
  trait.assoc_type @Elem = i64
}

trait.trait private @T(%self: !trait.claim<@T[!S]>) -> !trait.claim<!trait.proj<@Sib[!S], "Elem"> = f32> {
  trait.method @id(!S) -> !S
}

// expected-error @below {{returns '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>' for requirement 0, which trait '@T' states as '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>'}}
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {
  trait.method @id(%x: i64) -> i64 {
    trait.return %x : i64
  }
  %e = trait.witness proj_resolve !trait.proj<@Sib[i64], "Elem"> resolves i64 by @Sib_i64 : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>
  trait.return %e : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>
}
