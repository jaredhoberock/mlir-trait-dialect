// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A call spells its template's `!Y` as the ground projection
// `Producer[i64]::Item` twice: in the `@Has` claim, a coercion of the witness of
// `@Has[i32]` citing the projection's binding, and in the equality operand,
// whose endpoints nothing resolves. Where the call is lowered the coercion has
// settled to the witness of `@Has_i32_p` respelled at the projection. The
// clone's claim parameters rebind the variables inside their predicates and
// nothing else, so they match the operands only when `!Y` is read off the
// equality; the `@Has` position compares through selection, which reduces that
// spelling to `i32`. Reading `!Y` off `@Has` first bound it to `i32`, the
// clone's parameters differed from the operands, and the call was never
// lowered.
//
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

trait.trait private @Producer(%self: !trait.claim<@Producer[!trait.poly<0>]>) {
  trait.assoc_type @Item
}
trait.impl private @Producer_i64(%self: !trait.claim<@Producer[i64]>) {
  trait.assoc_type @Item = i32
}
trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) {}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {}
trait.proof private @Has_i32_p {
  %d = trait.derive @Has[i32] from @Has_i32 given()
  trait.return %d : !trait.claim<@Has[i32]>
}

func.func private @tpl(%h: !trait.claim<@Has[!trait.poly<1>]>,
                       %e: !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.poly<1>>) {
  return
}

// CHECK-LABEL: func.func @main
// CHECK: call @tpl_{{.*}}(%{{.*}}, %{{.*}}) : (!trait.claim<@Has[!trait.proj<@Producer[i64], "Item">] by @Has_i32_{{.*}}>, !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.proj<@Producer[i64], "Item">>) -> ()
func.func @main(%e: !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.proj<@Producer[i64], "Item">>) {
  %w = trait.witness @Has_i32_p for @Has[i32]
  %item = trait.witness proj_resolve !trait.proj<@Producer[i64], "Item"> resolves i32 by @Producer_i64 : !trait.claim<!trait.proj<@Producer[i64], "Item"> = i32>
  %h = trait.coerce %w : !trait.claim<@Has[i32] by @Has_i32_p> to !trait.claim<@Has[!trait.proj<@Producer[i64], "Item">]> via (%item) : (!trait.claim<!trait.proj<@Producer[i64], "Item"> = i32>)
  trait.func.call @tpl(%h, %e)
    : (!trait.claim<@Has[!trait.proj<@Producer[i64], "Item">]>,
       !trait.claim<!trait.proj<@Producer[i64], "Item"> = !trait.proj<@Producer[i64], "Item">>) -> ()
  return
}
