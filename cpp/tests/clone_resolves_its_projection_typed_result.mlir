// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A template whose result type is a projection on its own claim parameter is
// cloned for a concrete claim: the binding grounds the projection, so the
// clone's result type, the value it returns and the method call producing it all
// land on the impl's associated type together, and the call is lowered to the
// impl's method.
//
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

trait.trait private @Producer[!trait.poly<0>] {
  trait.assoc_type @Item
  func.func private @make(!trait.poly<0>) -> !trait.proj<@Producer[!trait.poly<0>], "Item">
}
trait.impl private @Producer_i64 for @Producer[i64] {
  trait.assoc_type @Item = i32
  func.func @make(%x: i64) -> i32 {
    %c = arith.constant 0 : i32
    return %c : i32
  }
}
trait.proof private @Producer_i64_p proves @Producer_i64 for @Producer[i64] given []

func.func private @tpl(%claim: !trait.claim<@Producer[!trait.poly<0>]>, %v: !trait.poly<0>)
    -> !trait.proj<@Producer[!trait.poly<0>], "Item"> {
  %m = trait.method.call %claim @Producer[!trait.poly<0>]::@make(%v)
    : (!trait.poly<0>) -> !trait.proj<@Producer[!trait.poly<0>], "Item">
  return %m : !trait.proj<@Producer[!trait.poly<0>], "Item">
}

func.func @main(%v: i64) -> !trait.proj<@Producer[i64], "Item"> {
  %c = trait.witness @Producer_i64 for @Producer[i64]
  %r = trait.func.call @tpl(%c, %v) {type_params = [!trait.poly<0>], type_args = [i64]}
    : (!trait.claim<@Producer[i64] by @Producer_i64>, i64) -> !trait.proj<@Producer[i64], "Item">
  return %r : !trait.proj<@Producer[i64], "Item">
}

// CHECK: func.func {{.*}}@tpl_
// CHECK-SAME: -> i32
// CHECK: call @Producer_i64_make
// CHECK: return %{{.*}} : i32
// CHECK-LABEL: func.func @main
// CHECK: call @tpl_
