// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s --check-prefix=INSTANCE --implicit-check-not=error:
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A proof may spell a premise through a ground projection: @p cites @Zero_f32
// for Zero[Tensor[i32]::Element]. The method instance cut at that proof
// clones the premise into its entry spelled as the rest of the instance is,
// with the projection resolved, so the call it reaches takes the claim the
// callee's instance declares.

// INSTANCE:      func.func private @A_gen_{{.*}}_a(%{{.*}}: !trait.claim<@A[f32] by @p>) -> i64 {
// INSTANCE-NEXT:   %[[ZERO:.*]] = trait.witness @Zero_f32 for @Zero[f32]
// INSTANCE-NEXT:   call @Zero_f32_{{.*}}_zero(%[[ZERO]]) : (!trait.claim<@Zero[f32] by @Zero_f32>) -> i64

// CHECK: {{^}}9{{$}}

!P = !trait.poly<0>
trait.trait private @Tensor(%self: !trait.claim<@Tensor[!P]>) { trait.assoc_type @Element }
trait.trait private @Zero(%self: !trait.claim<@Zero[!P]>) { trait.method @zero() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!P]>) { trait.method @a() -> i64 }
trait.impl private @Tensor_i32(%self: !trait.claim<@Tensor[i32]>) { trait.assoc_type @Element = f32 }
trait.impl private @Zero_f32(%self: !trait.claim<@Zero[f32]>) {
  trait.method @zero() -> i64 {
    %c = arith.constant 9 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_gen(%self: !trait.claim<@A[!P]>, %zero: !trait.claim<@Zero[!P]>) {
  trait.method @a() -> i64 {
    %r = trait.method.call %zero @Zero[!P]::@zero() : () -> i64
    trait.return %r : i64
  }
}
trait.proof private @p {
  %z = trait.witness @Zero_f32 for @Zero[!trait.proj<@Tensor[i32], "Element">]
  %d = trait.derive @A[f32] from @A_gen given(%z) : (!trait.claim<@Zero[!trait.proj<@Tensor[i32], "Element">] by @Zero_f32>)
  trait.return %d : !trait.claim<@A[f32]>
}
func.func @main() -> i64 {
  %w = trait.witness @p for @A[f32]
  %r = trait.method.call %w @A[f32]::@a() : () -> i64 by @p
  return %r : i64
}
