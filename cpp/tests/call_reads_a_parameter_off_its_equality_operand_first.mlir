// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A call spells its template's `!Y` twice: resolved, in the `@Has[i32]` claim,
// and as the ground projection `Producer[i64]::Item`, in the equality operand,
// whose endpoints nothing resolves. The clone's equality parameter rebinds the
// variables inside its endpoints and nothing else, so it matches the operand only
// when `!Y` is read off the equality; the `@Has` position compares through
// normalization, which reduces that spelling to `i32`. Reading `!Y` off `@Has`
// first bound it to `i32`, the clone's equality parameter differed from the
// operand, and the call was never lowered.
//
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

trait.trait private @Producer[!trait.poly<0>] {
  trait.assoc_type @Item
}
trait.impl private @Producer_i64 for @Producer[i64] {
  trait.assoc_type @Item = i32
}
trait.trait private @Has[!trait.poly<0>] {}
trait.impl private @Has_i32 for @Has[i32] {}
trait.proof private @Has_i32_p proves @Has_i32 for @Has[i32] given []

func.func private @tpl(%h: !trait.claim<@Has[!trait.poly<1>]>,
                       %e: !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.poly<1>>) {
  return
}

// CHECK-LABEL: func.func @main
// CHECK: call @tpl_{{.*}}(%{{.*}}, %{{.*}}) : (!trait.claim<@Has[i32] by @Has_i32_p>, !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.proj<@Producer[i64], "Item">>) -> ()
func.func @main(%e: !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.proj<@Producer[i64], "Item">>) {
  %h = trait.witness @Has_i32_p for @Has[i32]
  trait.func.call @tpl(%h, %e)
    : (!trait.claim<@Has[i32] by @Has_i32_p>,
       !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.proj<@Producer[i64], "Item">>) -> ()
  return
}
