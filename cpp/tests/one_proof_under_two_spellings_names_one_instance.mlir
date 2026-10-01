// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=INSTANCES
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One proof is supplied under two spellings of its application: at @Mark[i64]
// or @Tr[i64], and coerced to spell i64 as @Fold[i32]::Item, a projection only
// the stage's record settles (@Fold_gen binds Item through @Tensor[i32]'s
// Element, which @Tensor_i32 binds to i64). Evidence is read at the spelling
// the instance is stamped in, so each pair of calls supplies one piece of
// evidence and reaches one instance: one for a claim argument, and one for a
// receiver, whose citation of @Tr_gen's where clause reads @P's subproof by
// position at the spelling @P states it for.

// INSTANCES-COUNT-1: func.func private @Tr_i64{{[_a-z0-9]*}}_run(
// INSTANCES-NOT: func.func private @Tr_i64{{[_a-z0-9]*}}_run(
// INSTANCES-COUNT-1: func.func private @Tr_gen{{[_a-z0-9]*}}_get(
// INSTANCES-NOT: func.func private @Tr_gen{{[_a-z0-9]*}}_get(

// CHECK: {{^}}28{{$}}

!A = !trait.poly<0>
!B = !trait.poly<1>
trait.trait private @Tensor[!A] { trait.assoc_type @Element }
trait.impl private @Tensor_i32 for @Tensor[i32] { trait.assoc_type @Element = i64 }
trait.trait private @Vec[!A] {}
trait.impl private @Vec_i32 for @Vec[i32] {}
trait.trait private @Fold[!A] { trait.assoc_type @Item }
trait.impl private @Fold_gen for @Fold[!B] where [@Vec[!B]] {
  trait.assoc_type @Item = !trait.proj<@Tensor[!B], "Element">
}
trait.trait private @Mark[!A] { func.func private @value() -> i64 }
trait.impl private @Mark_i64 for @Mark[i64] {
  func.func @value() -> i64 {
    %v = arith.constant 7 : i64
    return %v : i64
  }
}
trait.trait private @Tr[!A] {
  func.func private @run(!trait.claim<@Mark[!A]>) -> i64
}
trait.impl private @Tr_i64 for @Tr[i64] {
  func.func @run(%m: !trait.claim<@Mark[i64]>) -> i64 {
    %v = trait.method.call %m @Mark[i64]::@value() : () -> i64
    return %v : i64
  }
}
trait.trait private @Get[!A] { func.func private @get() -> i64 }
trait.impl private @Tr_gen for @Get[!B] where [@Mark[!B]] {
  func.func @get() -> i64 {
    %m = trait.assume 0 : !trait.claim<@Mark[!B]>
    %v = trait.method.call %m @Mark[!B]::@value() : () -> i64
    return %v : i64
  }
}
trait.proof private @P proves @Tr_gen[!B = i64] for @Get[i64] given [@Mark_i64]
func.func @main() -> i64 {
  %vec = trait.witness @Vec_i32 for @Vec[i32]
  %item = trait.witness proj_resolve !trait.proj<@Fold[i32], "Item"> resolves !trait.proj<@Tensor[i32], "Element"> by @Fold_gen[!B = i32] given(%vec) : (!trait.claim<@Vec[i32] by @Vec_i32>) : !trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>
  %elem = trait.witness proj_resolve !trait.proj<@Tensor[i32], "Element"> resolves i64 by @Tensor_i32 : !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>

  %tr = trait.witness @Tr_i64 for @Tr[i64]
  %m = trait.witness @Mark_i64 for @Mark[i64]
  %m_spelled = trait.coerce %m : !trait.claim<@Mark[i64] by @Mark_i64> to !trait.claim<@Mark[!trait.proj<@Fold[i32], "Item">] by @Mark_i64> via (%item, %elem) : (!trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>, !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>)
  %a = trait.method.call %tr @Tr[i64]::@run(%m_spelled) : (!trait.claim<@Mark[!trait.proj<@Fold[i32], "Item">] by @Mark_i64>) -> i64 by @Tr_i64
  %b = trait.method.call %tr @Tr[i64]::@run(%m) : (!trait.claim<@Mark[i64] by @Mark_i64>) -> i64 by @Tr_i64

  %p = trait.witness @P for @Get[i64]
  %p_spelled = trait.coerce %p : !trait.claim<@Get[i64] by @P> to !trait.claim<@Get[!trait.proj<@Fold[i32], "Item">] by @P> via (%item, %elem) : (!trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>, !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>)
  %c = trait.method.call %p_spelled @Get[!trait.proj<@Fold[i32], "Item">]::@get() : () -> i64 by @P
  %d = trait.method.call %p @Get[i64]::@get() : () -> i64 by @P

  %ab = arith.addi %a, %b : i64
  %cd = arith.addi %c, %d : i64
  %s = arith.addi %ab, %cd : i64
  return %s : i64
}
