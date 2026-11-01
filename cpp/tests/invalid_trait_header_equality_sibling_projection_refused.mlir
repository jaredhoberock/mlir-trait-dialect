// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A trait's header requires a sibling-projection equality (@Sib[Self]::Elem =
// f32). The requirement is evidence the impl returns, and @Sib_i64 binds
// @Sib[i64]::Elem to i64, so no evidence makes the two endpoints one type. An
// impl alleging the equality is refused where its allegation is decided: the
// instance of a member that projects the requirement off the impl's own claim,
// where the projection is replaced by the allegation, located as the impl
// wrote it.
// The impl that satisfies such a requirement returns the witness reducing the
// endpoint; impl_satisfies_trait_equality_requirement.mlir is that row.

!S = !trait.poly<0>

trait.trait private @Sib(!trait.claim<@Sib[!S]>) assoc {Elem = 0}

trait.impl private @Sib_i64(%self: !trait.claim<@Sib[i64]>) bindings {Elem = i64} {
}

trait.trait private @T(!trait.claim<@T[!S]>) -> !trait.claim<!trait.proj<@Sib[!S], "Elem"> = f32> members [@T.get]
func.func private @T.get(!trait.claim<@T[!S]>, !S) -> !trait.proj<@Sib[!S], "Elem">

trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) members [@T.get -> @T_i64.get] {
  // expected-error @below {{alleges '!trait.proj<@Sib[i64], "Elem">' = 'f32', and impl selection resolves its sides to 'i64' and 'f32'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>' after instantiate-monomorphs}}
  %alleged = trait.allege !trait.proj<@Sib[i64], "Elem"> = f32
  trait.return %alleged : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>
}
func.func private @T_i64.get(%self: !trait.claim<@T[i64]>, %x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
  %c = arith.constant 1.0 : f32
  %e = trait.project %self[0] : !trait.claim<@T[i64]> -> !trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>
  %v = trait.coerce %c : f32 to !trait.proj<@Sib[i64], "Elem"> via (%e) : (!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>)
  return %v : !trait.proj<@Sib[i64], "Elem">
}

func.func @main(%x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
  %w = trait.witness @T_i64 for @T[i64]
  %r = trait.func.call @T.get(%w, %x) : (!trait.claim<@T[i64] by @T_i64>, i64) -> !trait.proj<@Sib[i64], "Elem">
  return %r : !trait.proj<@Sib[i64], "Elem">
}

// -----

// The same requirement with the impl returning the evidence it has: @Sib_i64's
// resolution makes @Sib[i64]::Elem i64, which is not the requirement, so the
// return is refused where it is written.

!S = !trait.poly<0>

trait.trait private @Sib(!trait.claim<@Sib[!S]>) assoc {Elem = 0}

trait.impl private @Sib_i64(%self: !trait.claim<@Sib[i64]>) bindings {Elem = i64} {
}

trait.trait private @T(!trait.claim<@T[!S]>) -> !trait.claim<!trait.proj<@Sib[!S], "Elem"> = f32> members [@T.id]
func.func private @T.id(!trait.claim<@T[!S]>, !S) -> !S

// expected-error @below {{returns '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>' for requirement 0, which trait '@T' states as '!trait.claim<!trait.proj<@Sib[i64], "Elem"> = f32>'}}
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) members [@T.id -> @T_i64.id] {
  %e = trait.witness proj_resolve !trait.proj<@Sib[i64], "Elem"> resolves i64 by @Sib_i64 : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>
  trait.return %e : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i64>
}
func.func private @T_i64.id(%self: !trait.claim<@T[i64]>, %x: i64) -> i64 {
  return %x : i64
}
