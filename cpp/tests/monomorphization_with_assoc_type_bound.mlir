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

trait.trait private @Printable(!trait.claim<@Printable[!S]>) members [@Printable.print]
func.func private @Printable.print(!trait.claim<@Printable[!S]>, !S) -> i32

trait.impl private @Printable_impl_i64(%self_claim: !trait.claim<@Printable[i64]>) members [@Printable.print -> @Printable_impl_i64.print] {
}
func.func private @Printable_impl_i64.print(%self_claim: !trait.claim<@Printable[i64]>, %self: i64) -> i32 {
  %c = arith.trunci %self : i64 to i32
  return %c : i32
}

trait.trait private @Container(!trait.claim<@Container[!S]>) -> !trait.claim<@Printable[!trait.proj<@Container[!S], "Elem">]> assoc {Elem = 0} members [@Container.first]
func.func private @Container.first(!trait.claim<@Container[!S]>, !S) -> !trait.proj<@Container[!S], "Elem">

trait.impl private @Container_impl_i32(%self_claim: !trait.claim<@Container[i32]>) bindings {Elem = i64} members [@Container.first -> @Container_impl_i32.first] {
  %req0 = trait.allege @Printable[!trait.proj<@Container[i32], "Elem">]
  trait.return %req0 : !trait.claim<@Printable[!trait.proj<@Container[i32], "Elem">]>
}
func.func private @Container_impl_i32.first(%self_claim: !trait.claim<@Container[i32]>, %self: i32) -> i64 {
  %c = arith.extsi %self : i32 to i64
  return %c : i64
}

// Wraps @Container: @Wrapper[T] requires @Container[T]
trait.trait private @Wrapper(!trait.claim<@Wrapper[!S]>) members [@Wrapper.get]
func.func private @Wrapper.get(!trait.claim<@Wrapper[!S]>, !S) -> i32

!Wi = !trait.poly<1>
trait.impl private @Wrapper_impl(%self_claim: !trait.claim<@Wrapper[!trait.poly<0>]>, %container_1: !trait.claim<@Container[!trait.poly<0>]>) members [@Wrapper.get -> @Wrapper_impl.get] {
}
func.func private @Wrapper_impl.get(%self_claim: !trait.claim<@Wrapper[!trait.poly<0>]>, %container_1: !trait.claim<@Container[!trait.poly<0>]>, %self: !trait.poly<0>) -> i32 {
  // use the @Container[T] assumption to call @first, then @Printable to print
  %elem = trait.func.call @Container.first(%container_1, %self) : (!trait.claim<@Container[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Container[!trait.poly<0>], "Elem">

  // @Container's sole requirement is the @Printable obligation
  %printable = trait.project %container_1[0]
    : !trait.claim<@Container[!trait.poly<0>]>
    -> !trait.claim<@Printable[!trait.proj<@Container[!trait.poly<0>], "Elem">]>

  %result = trait.func.call @Printable.print(%printable, %elem) : (!trait.claim<@Printable[!trait.proj<@Container[!trait.poly<0>], "Elem">]>, !trait.proj<@Container[!trait.poly<0>], "Elem">) -> i32

  return %result : i32
}

// Monomorphic call site: @Wrapper[i32] -> @Container[i32] -> @Printable[i64].

// CHECK-LABEL: func.func @caller
// CHECK-NOT: trait.allege
// CHECK-NOT: !trait.proj
// CHECK: call @Wrapper_impl.get_{{.*}}
// CHECK: return
func.func @caller() -> i32 {
  %w = trait.allege @Wrapper[i32]
  %x = arith.constant 42 : i32
  %res = trait.func.call @Wrapper.get(%w, %x) : (!trait.claim<@Wrapper[i32]>, i32) -> i32
  return %res : i32
}
