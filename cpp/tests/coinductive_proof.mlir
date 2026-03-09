// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

// Tests coinductive proof resolution.
//
// @Rec has a self-referencing where clause: @Rec[!trait.proj<@Rec[!S], "Sub">].
// When proving @Rec[i64], the obligation is @Rec[proj(@Rec[i64], "Sub")]
// which resolves to @Rec[i64] — the same claim.  Without coinductive cycle
// detection this diverges.

!S = !trait.poly<0>
trait.trait @Rec[!S] where [@Rec[!trait.proj<@Rec[!S], "Sub">]] {
  trait.assoc_type @Sub
  func.func private @id(!S) -> !S
}

trait.impl @Rec_i64 for @Rec[i64] {
  trait.assoc_type @Sub = i64
  func.func private @id(%x: i64) -> i64 { return %x : i64 }
}

trait.impl @Rec_unit for @Rec[tuple<>] {
  trait.assoc_type @Sub = tuple<>
  func.func private @id(%x: tuple<>) -> tuple<> { return %x : tuple<> }
}

func.func @test_coinductive(%x: i64) -> i64 {
  // CHECK: trait.witness @Rec_i64_p for @Rec[i64]
  %c = trait.allege @Rec[i64]
  %res = trait.method.call %c @Rec[i64]::@id(%x) : (i64) -> i64
  return %res : i64
}

func.func @test_coinductive_unit(%x: tuple<>) -> tuple<> {
  // CHECK: trait.witness @Rec_unit_p for @Rec[tuple<>]
  %c = trait.allege @Rec[tuple<>]
  %res = trait.method.call %c @Rec[tuple<>]::@id(%x) : (tuple<>) -> tuple<>
  return %res : tuple<>
}

// The coinductive obligation resolves back to the same claim, so the proof
// references itself as the subproof.
// CHECK: trait.proof @Rec_unit_p proves @Rec_unit for @Rec[tuple<>] given [@Rec_unit_p]
// CHECK: trait.proof @Rec_i64_p proves @Rec_i64 for @Rec[i64] given [@Rec_i64_p]
