// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Evidence that returns to its own question through another impl: @A_i32's
// evidence for @B[i32] is @C_i32's, and @C_i32's is @A_i32's. Each impl reads
// another's return, so each verifies; a projection that reads the cycle is
// replaced by @A_i32's return, which projects @C_i32's, which projects
// @A_i32's again, and the stage's rewrite budget is what stops it.

// CHECK: error: instantiate-monomorphs did not converge: rewrite budget exceeded

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %c = trait.witness @C_i32 for @C[i32]
  %b = trait.project %c[0] : !trait.claim<@C[i32] by @C_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  %v = trait.method.call %b @B[i32]::@v() : () -> i64
  return %v : i64
}

// -----

// Evidence that returns to its own question through two requirements of one
// impl: requirement 0 is requirement 1 and requirement 1 is requirement 0, both
// read off a witness of the impl itself, and the impl is refused where it is
// declared.

// CHECK: error: 'trait.impl' op returns evidence for requirement 0 that projects the impl's own application

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> (!trait.claim<@B[!T]>, !trait.claim<@B[!T]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %x = trait.project %a[1] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  %y = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %x, %y : !trait.claim<@B[i32]>, !trait.claim<@B[i32]>
}

// -----

// The first cycle read inside an instance: @callee's projection off its proven
// parameter is inlined as the instance is cut, then the projection @A_i32's
// return computes, then @C_i32's, around the cycle. The cut inlines at most as
// many rounds as the instantiation limit and leaves the rest to the stage's
// rewrite budget, which refuses it.

// CHECK: error: instantiate-monomorphs did not converge: rewrite budget exceeded

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %c = trait.witness @C_i32 for @C[i32]
  %b = trait.project %c[0] : !trait.claim<@C[i32] by @C_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
func.func private @callee(%c: !trait.claim<@A[!T]>) -> i64 {
  %b = trait.project %c[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
  %v = trait.method.call %b @B[!T]::@v() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %a = trait.witness @A_i32 for @A[i32]
  %v = trait.func.call @callee(%a) : (!trait.claim<@A[i32] by @A_i32>) -> i64
  return %v : i64
}
