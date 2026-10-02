// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Two modules spell one claim identically and mean two different proofs of it:
// @B[i32] by @p derives @B_impl, which returns @A_top for its requirement out
// here and @A_inner in there. What a proof's requirement evaluates to is a fact
// about the module it was read from, so neither module's answers for the
// other's.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {}
trait.impl private @A_top(%self: !trait.claim<@A[i32]>) {}
trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) {
  %a = trait.witness @A_top for @A[i32]
  trait.return %a : !trait.claim<@A[i32] by @A_top>
}
trait.proof private @p {
  %d = trait.derive @B[i32] from @B_impl given()
  trait.return %d : !trait.claim<@B[i32]>
}

func.func private @callee(%c: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  %a = trait.project %c[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
  return %a : !trait.claim<@A[!trait.poly<0>]>
}

// CHECK: func.func private @[[OUTER:callee_h[0-9a-f]+]]()
// CHECK: func.func @main
// CHECK: call @[[OUTER]]
func.func @main() -> !trait.claim<@A[i32] by @A_top> {
  %w = trait.witness @p for @B[i32]
  %r = trait.func.call @callee(%w) : (!trait.claim<@B[i32] by @p>) -> !trait.claim<@A[i32] by @A_top>
  return %r : !trait.claim<@A[i32] by @A_top>
}

// CHECK: module @inner
module @inner {
  trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
  trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {}
  trait.impl private @A_inner(%self: !trait.claim<@A[i32]>) {}
  trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) {
    %a = trait.witness @A_inner for @A[i32]
    trait.return %a : !trait.claim<@A[i32] by @A_inner>
  }
  trait.proof private @p {
    %d = trait.derive @B[i32] from @B_impl given()
    trait.return %d : !trait.claim<@B[i32]>
  }

  func.func private @callee(%c: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
    %a = trait.project %c[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
    return %a : !trait.claim<@A[!trait.poly<0>]>
  }

  // CHECK: func.func private @[[INNER:callee_h[0-9a-f]+]]()
  // CHECK: func.func @main
  // CHECK: call @[[INNER]]
  func.func @main() -> !trait.claim<@A[i32] by @A_inner> {
    %w = trait.witness @p for @B[i32]
    %r = trait.func.call @callee(%w) : (!trait.claim<@B[i32] by @p>) -> !trait.claim<@A[i32] by @A_inner>
    return %r : !trait.claim<@A[i32] by @A_inner>
  }
}
