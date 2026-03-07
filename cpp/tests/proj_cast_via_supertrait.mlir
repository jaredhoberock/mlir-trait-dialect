// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Transliterated from todolang-main test 14_29:
//   trait Base { type Assoc; }
//   trait Child: Base {}
//   impl Base for Number { type Assoc = Bool; }
//   impl Child for Number {}
//   fn foo<T: Child>(arg: T, value: T::Assoc) -> T::Assoc { value }
//   assert foo(0, true);
//
// The call site uses `trait.proj.cast` to convert the concrete `i1` argument
// into a projection type so that the verifier can unify it with the polymorphic
// formal type `!trait.proj<@Base[!T], "Assoc">` without impl resolution.

!T = !trait.poly<0>

trait.trait @Base[!T] {
  trait.assoc_type @Assoc
}

trait.trait @Child[!T] where [@Base[!T]] {
}

trait.impl @Base_impl for @Base[i64] {
  trait.assoc_type @Assoc = i1
}

trait.impl @Child_impl for @Child[i64] {
}

trait.proof @Base_proof proves @Base_impl for @Base[i64] given []
trait.proof @Child_proof proves @Child_impl for @Child[i64] given [
  @Base_proof
]

// A polymorphic function whose parameter and return types are projections
// of the supertrait @Base, accessed through the @Child bound.
func.func @foo(
    %arg: !T,
    %value: !trait.proj<@Base[!T], "Assoc">,
    %claim: !trait.claim<@Child[!T]>
) -> !trait.proj<@Base[!T], "Assoc"> {
  return %value : !trait.proj<@Base[!T], "Assoc">
}

// CHECK-LABEL: func.func @main
// CHECK: call @foo_
// CHECK-SAME: (i64, i1) -> i1
// CHECK: return %{{.*}} : i1
func.func @main() -> i1 {
  %arg = arith.constant 0 : i64
  %val = arith.constant true

  %child_claim = trait.witness @Child_proof for @Child[i64]
  %base_claim = trait.witness @Base_proof for @Base[i64]

  // Cast the concrete i1 value to the projection type via claim
  %val_proj = trait.proj.cast %val, %base_claim
    : i1 to !trait.proj<@Base[i64], "Assoc"> claim !trait.claim<@Base[i64] by @Base_proof>

  %result = trait.func.call @foo(%arg, %val_proj, %child_claim)
    : (i64, !trait.proj<@Base[i64], "Assoc">,
       !trait.claim<@Child[i64] by @Child_proof>) -> i1

  return %result : i1
}
