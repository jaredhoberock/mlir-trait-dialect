// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A premise whose reading carries no type variable is decided here even where
// it spells a projection nothing resolves: no instance of anything moves that
// spelling, so an impl whose premise it leaves unequal does not apply. Nothing
// implements @Foo, so @Foo[i8]::Out stands as written and is not i64.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.trait private @Tensor[!trait.poly<0>] {}
trait.trait private @Vector[!trait.poly<0>] { func.func private @v() -> i64 }
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Foo[!trait.poly<0>], "Out"> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @Tensor_i8 for @Tensor[i8] {}
// expected-error @below {{impl '@Vector_blanket' applies where '!trait.proj<@Foo[!trait.poly<0>], "Out">' = 'i64', and nothing here makes '!trait.proj<@Foo[i8], "Out">' and 'i64' one type at '!trait.claim<@Vector[i8] by @p>'}}
trait.proof private @p proves @Vector_blanket for @Vector[i8] given [@Tensor_i8]

// -----

// The same rule where the reading is unresolved because two impls of @Foo bind
// @Foo[i8] rather than none. Both bind @Out to i32, so the impl that applies at
// i8 is @Vector_b; the proof of @Vector_a is a proof of the impl whose premise
// is false, and a witness of it would run @Vector_a's method.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = i32 }
trait.impl private @Foo_i8 for @Foo[i8] { trait.assoc_type @Out = i32 }
trait.trait private @Tensor[!trait.poly<0>] {}
trait.trait private @Vector[!trait.poly<0>] { func.func private @v() -> i64 }
trait.impl private @Vector_a for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Foo[!trait.poly<0>], "Out"> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
trait.impl private @Vector_b for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Foo[!trait.poly<0>], "Out"> = i32] {
  func.func @v() -> i64 {
    %c = arith.constant 2 : i64
    return %c : i64
  }
}
trait.impl private @Tensor_i8 for @Tensor[i8] {}
// expected-error @below {{impl '@Vector_a' applies where '!trait.proj<@Foo[!trait.poly<0>], "Out">' = 'i64', and nothing here makes '!trait.proj<@Foo[i8], "Out">' and 'i64' one type at '!trait.claim<@Vector[i8] by @p>'}}
trait.proof private @p proves @Vector_a for @Vector[i8] given [@Tensor_i8]
