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

trait.trait @Printable[!S] {
  func.func private @print(!S) -> i32
}

trait.impl @Printable_impl_i64 for @Printable[i64] {
  func.func @print(%self: i64) -> i32 {
    %c = arith.trunci %self : i64 to i32
    return %c : i32
  }
}

trait.trait @Container[!S] where [@Printable[!trait.proj<@Container[!S], "Elem">]] {
  trait.assoc_type @Elem
  func.func private @first(!S) -> !trait.proj<@Container[!S], "Elem">
}

trait.impl @Container_impl_i32 for @Container[i32] {
  trait.assoc_type @Elem = i64
  func.func @first(%self: i32) -> i64 {
    %c = arith.extsi %self : i32 to i64
    return %c : i64
  }
}

// Wraps @Container: @Wrapper[T] requires @Container[T]
trait.trait @Wrapper[!S] {
  func.func private @get(!S) -> i32
}

!Wi = !trait.poly<1>
trait.impl @Wrapper_impl for @Wrapper[!Wi] where [
  @Container[!Wi]
] {
  func.func @get(%self: !Wi) -> i32 {
    // use the @Container[T] assumption to call @first, then @Printable to print
    %container = trait.assume @Container[!Wi]
    %elem = trait.method.call %container @Container[!Wi]::@first(%self)
      : (!Wi) -> !trait.proj<@Container[!Wi], "Elem">

    // project the @Printable obligation from @Container
    %printable = trait.project %container
      : @Container[!Wi]
      to @Printable[!trait.proj<@Container[!Wi], "Elem">]

    %result = trait.method.call %printable @Printable[!trait.proj<@Container[!Wi], "Elem">]::@print(%elem)
      : (!trait.proj<@Container[!Wi], "Elem">) -> i32

    return %result : i32
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
