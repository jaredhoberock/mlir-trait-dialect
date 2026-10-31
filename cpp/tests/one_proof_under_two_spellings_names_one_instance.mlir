// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=INSTANCES
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One proof is supplied under two spellings of its application: at @Mark[i64]
// or @Get[i64], and coerced to spell i64 as @Fold[i32]::Item, a projection only
// the stage's record settles (@Fold_gen binds Item through @Tensor[i32]'s
// Element, which @Tensor_i32 binds to i64). A coercion of a proof settles to
// that proof respelled at the coercion's spelling, and evidence is read at the
// spelling the instance's parameter has, so each pair of calls supplies one
// piece of evidence and reaches one instance: one for a claim argument coerced
// back to the written form the member declares, and one for a receiver, whose
// proof respelled at @Tr_gen's header is @P again.

// INSTANCES-COUNT-1: func.func private @Tr_i64.run_{{h[0-9a-f]+}}(
// INSTANCES-NOT: func.func private @Tr_i64.run_{{h[0-9a-f]+}}(
// INSTANCES-COUNT-1: func.func private @Tr_gen.get_{{h[0-9a-f]+}}(
// INSTANCES-NOT: func.func private @Tr_gen.get_{{h[0-9a-f]+}}(

// CHECK: {{^}}28{{$}}

!A = !trait.poly<0>
!B = !trait.poly<1>
trait.trait private @Tensor(!trait.claim<@Tensor[!A]>) assoc {Element = 0}
trait.impl private @Tensor_i32(%self: !trait.claim<@Tensor[i32]>) bindings {Element = i64} {
}
trait.trait private @Vec(!trait.claim<@Vec[!A]>)
trait.impl private @Vec_i32(%self: !trait.claim<@Vec[i32]>) {
}
trait.trait private @Fold(!trait.claim<@Fold[!A]>) assoc {Item = 0}
trait.impl private @Fold_gen(%self: !trait.claim<@Fold[!trait.poly<0>]>, %vec: !trait.claim<@Vec[!trait.poly<0>]>) bindings {Item = !trait.proj<@Tensor[!trait.poly<0>], "Element">} {
}
trait.trait private @Mark(!trait.claim<@Mark[!A]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!A]>) -> i64
trait.impl private @Mark_i64(%self: !trait.claim<@Mark[i64]>) members [@Mark.value -> @Mark_i64.value] {
}
func.func private @Mark_i64.value(%self: !trait.claim<@Mark[i64]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.trait private @Tr(!trait.claim<@Tr[!A]>) members [@Tr.run]
func.func private @Tr.run(!trait.claim<@Tr[!A]>, !trait.claim<@Mark[!A]>) -> i64
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) members [@Tr.run -> @Tr_i64.run] {
}
func.func private @Tr_i64.run(%self: !trait.claim<@Tr[i64]>, %m: !trait.claim<@Mark[i64]>) -> i64 {
  %v = trait.func.call @Mark.value(%m) : (!trait.claim<@Mark[i64]>) -> i64
  return %v : i64
}
trait.trait private @Get(!trait.claim<@Get[!A]>) members [@Get.get]
func.func private @Get.get(!trait.claim<@Get[!A]>) -> i64
trait.impl private @Tr_gen(%self: !trait.claim<@Get[!trait.poly<0>]>, %mark: !trait.claim<@Mark[!trait.poly<0>]>) members [@Get.get -> @Tr_gen.get] {
}
func.func private @Tr_gen.get(%self: !trait.claim<@Get[!trait.poly<0>]>, %mark: !trait.claim<@Mark[!trait.poly<0>]>) -> i64 {
  %v = trait.func.call @Mark.value(%mark) : (!trait.claim<@Mark[!trait.poly<0>]>) -> i64
  return %v : i64
}
trait.proof private @P {
  %p0 = trait.witness @Mark_i64 for @Mark[i64]
  %d = trait.derive @Get[i64] from @Tr_gen[i64] given(%p0) : (!trait.claim<@Mark[i64] by @Mark_i64>)
  trait.return %d : !trait.claim<@Get[i64]>
}
func.func @main() -> i64 {
  %vec = trait.witness @Vec_i32 for @Vec[i32]
  %item = trait.witness proj_resolve !trait.proj<@Fold[i32], "Item"> resolves !trait.proj<@Tensor[i32], "Element"> by @Fold_gen[i32] given(%vec) : (!trait.claim<@Vec[i32] by @Vec_i32>) : !trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>
  %elem = trait.witness proj_resolve !trait.proj<@Tensor[i32], "Element"> resolves i64 by @Tensor_i32 : !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>

  %tr = trait.witness @Tr_i64 for @Tr[i64]
  %m = trait.witness @Mark_i64 for @Mark[i64]
  %m_spelled = trait.coerce %m : !trait.claim<@Mark[i64] by @Mark_i64> to !trait.claim<@Mark[!trait.proj<@Fold[i32], "Item">]> via (%item, %elem) : (!trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>, !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>)
  %m_declared = trait.coerce %m_spelled : !trait.claim<@Mark[!trait.proj<@Fold[i32], "Item">]> to !trait.claim<@Mark[i64]> via (%item, %elem) : (!trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>, !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>)
  %a = trait.func.call @Tr.run(%tr, %m_declared) : (!trait.claim<@Tr[i64] by @Tr_i64>, !trait.claim<@Mark[i64]>) -> i64
  %b = trait.func.call @Tr.run(%tr, %m) : (!trait.claim<@Tr[i64] by @Tr_i64>, !trait.claim<@Mark[i64] by @Mark_i64>) -> i64

  %p = trait.witness @P for @Get[i64]
  %p_spelled = trait.coerce %p : !trait.claim<@Get[i64] by @P> to !trait.claim<@Get[!trait.proj<@Fold[i32], "Item">]> via (%item, %elem) : (!trait.claim<!trait.proj<@Fold[i32], "Item"> = !trait.proj<@Tensor[i32], "Element">>, !trait.claim<!trait.proj<@Tensor[i32], "Element"> = i64>)
  %c = trait.func.call @Get.get(%p_spelled) : (!trait.claim<@Get[!trait.proj<@Fold[i32], "Item">]>) -> i64
  %d = trait.func.call @Get.get(%p) : (!trait.claim<@Get[i64] by @P>) -> i64

  %ab = arith.addi %a, %b : i64
  %cd = arith.addi %c, %d : i64
  %s = arith.addi %ab, %cd : i64
  return %s : i64
}
