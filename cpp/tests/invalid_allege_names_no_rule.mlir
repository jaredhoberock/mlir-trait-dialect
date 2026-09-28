// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A rule is an attribute implementing RuleAttrInterface; a string names none.

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] {}
func.func @main() {
  %b = trait.allege @B[i32]
  // expected-error @below {{"not.a.rule" names no impl rule: a rule is an attribute implementing RuleAttrInterface}}
  %a = trait.allege @A[i32] by "not.a.rule" given(%b : !trait.claim<@B[i32]>)
  return
}

// -----

// Nor is a generated impl an instance of one.

trait.trait private @A[!trait.poly<0>] {}
// expected-error @below {{"not.a.rule" names no impl rule: a rule is an attribute implementing RuleAttrInterface}}
trait.impl private @A_i32 for @A[i32] by "not.a.rule" {}

// -----

// Premises are the premises of the rule an allegation names, and stand only
// with one.

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] {}
func.func @main(%b: !trait.claim<@B[i32]>) {
  // expected-error @below {{carries premises but names no rule they are the premises of}}
  %a = "trait.allege"(%b) : (!trait.claim<@B[i32]>) -> !trait.claim<@A[i32]>
  return
}
