// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// Only a call of a trait's member computing evidence is replaced by the body
// it reaches. A call of a free function returning a claim is cut into an
// instance like any other call of a template, and the instance returns the
// evidence its body reads by position.

// CHECK-LABEL: func.func private @f_h{{[0-9a-f]+}}(%{{.*}}: !trait.claim<@B[i32] by @B_i32>) -> !trait.claim<@A[i32] by @A_i32>
// CHECK-NEXT:    %[[A:.*]] = trait.witness @A_i32 for @A[i32]
// CHECK-NEXT:    return %[[A]]
// CHECK-LABEL: func.func @main
// CHECK:         call @f_h{{[0-9a-f]+}}(%{{.*}})

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>)
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]>
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  trait.return %a : !trait.claim<@A[i32] by @A_i32>
}
func.func private @f(%c: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  %a = trait.project %c[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
  return %a : !trait.claim<@A[!trait.poly<0>]>
}
func.func @main() -> !trait.claim<@A[i32] by @A_i32> {
  %w = trait.witness @B_i32 for @B[i32]
  %r = trait.func.call @f(%w) : (!trait.claim<@B[i32] by @B_i32>) -> !trait.claim<@A[i32] by @A_i32>
  return %r : !trait.claim<@A[i32] by @A_i32>
}
