// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A label names a position in the declaration that binds it, so it is
// non-negative: every other row in this suite spells `!trait.poly<0>`,
// `!trait.poly<1>` and so on. A negative label names no position and is refused
// where it is written, rather than standing for a variable no declaration binds.

// expected-error @below {{a !trait.poly label is non-negative; found -1}}
!T = !trait.poly<-1>
func.func @negative_label(%a: !T) -> !T {
  return %a : !T
}

// -----

// The refusal is on the spelling, so it reaches a label written inside a trait
// application as well as one written in a signature.

trait.trait private @Fold[!trait.poly<0>] {}

// expected-error @below {{a !trait.poly label is non-negative; found -3}}
func.func @negative_label_in_an_application(%c: !trait.claim<@Fold[!trait.poly<-3>]>) {
  return
}
