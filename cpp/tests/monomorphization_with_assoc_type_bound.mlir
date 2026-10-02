// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests that associated type bounds are correctly resolved during
// monomorphization, including through a polymorphic impl chain.
//
// @Container has an associated type @Elem with a requirement
// @Printable[proj<@Container[Self], "Elem">]. A polymorphic @Wrapper[T]
// impl requires @Container[T], inheriting the obligation chain.
// Monomorphizing @Wrapper[i32] must resolve the projection in @Container's
// requirement to @Printable[i64] so the sub-proof matches.

!S = !trait.poly<0>

trait.trait private @Printable(%self: !trait.claim<@Printable[!S]>) {
  trait.method @print(!S) -> i32
}

trait.impl private @Printable_impl_i64(%self_claim: !trait.claim<@Printable[i64]>) {
  trait.method @print(%self: i64) -> i32 {
    %c = arith.trunci %self : i64 to i32
    trait.return %c : i32
  }
}

trait.trait private @Container(%self: !trait.claim<@Container[!S]>) -> !trait.claim<@Printable[!trait.proj<@Container[!S], "Elem">]> {
  trait.assoc_type @Elem
  trait.method @first(!S) -> !trait.proj<@Container[!S], "Elem">
}

trait.impl private @Container_impl_i32(%self_claim: !trait.claim<@Container[i32]>) {
  trait.assoc_type @Elem = i64
  trait.method @first(%self: i32) -> i64 {
    %c = arith.extsi %self : i32 to i64
    trait.return %c : i64
  }
  %req0 = trait.allege @Printable[!trait.proj<@Container[i32], "Elem">]
  trait.return %req0 : !trait.claim<@Printable[!trait.proj<@Container[i32], "Elem">]>
}

// Wraps @Container: @Wrapper[T] requires @Container[T]
trait.trait private @Wrapper(%self: !trait.claim<@Wrapper[!S]>) {
  trait.method @get(!S) -> i32
}

!Wi = !trait.poly<1>
trait.impl private @Wrapper_impl(%self_claim: !trait.claim<@Wrapper[!Wi]>, %container_1: !trait.claim<@Container[!Wi]>) {
  trait.method @get(%self: !Wi) -> i32 {
    // use the @Container[T] assumption to call @first, then @Printable to print
    %elem = trait.method.call %container_1 @Container[!Wi]::@first(%self)
      : (!Wi) -> !trait.proj<@Container[!Wi], "Elem">

    // @Container's sole requirement is the @Printable obligation
    %printable = trait.project %container_1[0]
      : !trait.claim<@Container[!Wi]>
      -> !trait.claim<@Printable[!trait.proj<@Container[!Wi], "Elem">]>

    %result = trait.method.call %printable @Printable[!trait.proj<@Container[!Wi], "Elem">]::@print(%elem)
      : (!trait.proj<@Container[!Wi], "Elem">) -> i32

    trait.return %result : i32
  }
}

// Monomorphic call site: @Wrapper[i32] -> @Container[i32] -> @Printable[i64].

// CHECK-LABEL: func.func @caller
// CHECK-NOT: trait.allege
// CHECK-NOT: !trait.proj
// CHECK: call @Wrapper_impl_{{.*}}_get
// CHECK: return
func.func @caller() -> i32 {
  %w = trait.allege @Wrapper[i32]
  %x = arith.constant 42 : i32
  %res = trait.method.call %w @Wrapper[i32]::@get(%x)
    : (i32) -> i32
  return %res : i32
}
