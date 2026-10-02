// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --check-prefix=MONO

// Tests coinductive evidence.
//
// @Rec requires itself at its own associated type: @Rec[!trait.proj<@Rec[!S],
// "Sub">]. An impl binding Sub to its own self type proves that requirement
// with its own witness, which reads back to the claim the impl proves. Proving
// @Rec[i64] cites the impl without descending into the requirement, and the
// instances lower to the impls' methods.

!S = !trait.poly<0>
trait.trait private @Rec(%self: !trait.claim<@Rec[!S]>) -> !trait.claim<@Rec[!trait.proj<@Rec[!S], "Sub">]> {
  trait.assoc_type @Sub
  trait.method @id(!S) -> !S
}

trait.impl private @Rec_i64(%self: !trait.claim<@Rec[i64]>) {
  trait.assoc_type @Sub = i64
  trait.method @id(%x: i64) -> i64 { trait.return %x : i64 }
  %w = trait.witness @Rec_i64 for @Rec[i64]
  %eq = trait.witness proj_resolve !trait.proj<@Rec[i64], "Sub"> resolves i64 by @Rec_i64 : !trait.claim<!trait.proj<@Rec[i64], "Sub"> = i64>
  %req0 = trait.coerce %w : !trait.claim<@Rec[i64] by @Rec_i64> to !trait.claim<@Rec[!trait.proj<@Rec[i64], "Sub">] by @Rec_i64> via (%eq) : (!trait.claim<!trait.proj<@Rec[i64], "Sub"> = i64>)
  trait.return %req0 : !trait.claim<@Rec[!trait.proj<@Rec[i64], "Sub">] by @Rec_i64>
}

trait.impl private @Rec_unit(%self: !trait.claim<@Rec[tuple<>]>) {
  trait.assoc_type @Sub = tuple<>
  trait.method @id(%x: tuple<>) -> tuple<> { trait.return %x : tuple<> }
  %w = trait.witness @Rec_unit for @Rec[tuple<>]
  %eq = trait.witness proj_resolve !trait.proj<@Rec[tuple<>], "Sub"> resolves tuple<> by @Rec_unit : !trait.claim<!trait.proj<@Rec[tuple<>], "Sub"> = tuple<>>
  %req0 = trait.coerce %w : !trait.claim<@Rec[tuple<>] by @Rec_unit> to !trait.claim<@Rec[!trait.proj<@Rec[tuple<>], "Sub">] by @Rec_unit> via (%eq) : (!trait.claim<!trait.proj<@Rec[tuple<>], "Sub"> = tuple<>>)
  trait.return %req0 : !trait.claim<@Rec[!trait.proj<@Rec[tuple<>], "Sub">] by @Rec_unit>
}

func.func @test_coinductive(%x: i64) -> i64 {
  // CHECK: trait.witness @Rec_i64 for @Rec[i64]
  %c = trait.allege @Rec[i64]
  %res = trait.method.call %c @Rec[i64]::@id(%x) : (i64) -> i64
  return %res : i64
}

func.func @test_coinductive_unit(%x: tuple<>) -> tuple<> {
  // CHECK: trait.witness @Rec_unit for @Rec[tuple<>]
  %c = trait.allege @Rec[tuple<>]
  %res = trait.method.call %c @Rec[tuple<>]::@id(%x) : (tuple<>) -> tuple<>
  return %res : tuple<>
}

// MONO-LABEL: func.func @test_coinductive(%{{.*}}: i64) -> i64
// MONO: call @Rec_i64_{{.*}}_id
// MONO-LABEL: func.func @test_coinductive_unit
// MONO: call @Rec_unit_{{.*}}_id
