// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// One semantic application admits one proof symbol. Recording two claims for the
// same obligation is coherent exactly when they name the same proof; a second,
// different symbol for it is rejected outright by verifyEquivalentRecordedProof.
// This is the license for that now-permanent incoherent-rejection arm: two impls
// exist for @T[i64], so the proof symbol -- not the application -- is what makes
// a recording coherent.

trait.trait @T[!trait.poly<0>] {}

trait.impl @T_a for @T[i64] {}
trait.impl @T_b for @T[i64] {}

func.func private @callee(!trait.claim<@T[i64]>, !trait.claim<@T[i64]>)

// Coherent: both claims name the same proof symbol @T_a, so the second recording
// matches the first and is accepted.
func.func @coherent() {
  %a = trait.witness @T_a for @T[i64]
  trait.func.call @callee(%a, %a)
    : (!trait.claim<@T[i64] by @T_a>, !trait.claim<@T[i64] by @T_a>) -> ()
  return
}

// Incoherent: the two claims name different proof symbols for the one
// application, so the second recording conflicts and is rejected.
func.func @incoherent() {
  %a = trait.witness @T_a for @T[i64]
  %b = trait.witness @T_b for @T[i64]
  // expected-error @below {{inconsistent proof mapping}}
  trait.func.call @callee(%a, %b)
    : (!trait.claim<@T[i64] by @T_a>, !trait.claim<@T[i64] by @T_b>) -> ()
  return
}
