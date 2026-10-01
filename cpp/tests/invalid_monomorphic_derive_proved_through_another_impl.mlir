// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A derive stating its impl's arguments chose @Tr_any. At i32 that impl's
// premise does not hold, and impl selection proves the claim through @Tr_i32
// instead: two answers to one question, refused at the derive rather than
// settled silently by the second.

// CHECK: error: 'trait.derive' op derives '!trait.claim<@Tr[tuple<i32>]>' from impl @Tr_any, and impl selection proved it by @Tr_i32

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Mark[!T] {}
trait.trait private @Tr[!T] {
  trait.method @get(!T) -> i64
}

trait.impl private @Tr_any for @Tr[tuple<!U>] where [@Mark[!U]] {
  trait.method @get(%x: tuple<!U>) -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

trait.impl private @Tr_i32 for @Tr[tuple<i32>] {
  trait.method @get(%x: tuple<i32>) -> i64 {
    %c = arith.constant 2 : i64
    trait.return %c : i64
  }
}

func.func private @g(%x: tuple<!T>) -> i64 {
  %m = trait.allege @Mark[!T]
  %d = trait.derive @Tr[tuple<!T>] from @Tr_any[!U = !T] given(%m) : (!trait.claim<@Mark[!T]>)
  %r = trait.method.call %d @Tr[tuple<!T>]::@get(%x) : (tuple<!T>) -> i64
  return %r : i64
}

func.func @main(%x: tuple<i32>) -> i64 {
  %r = trait.func.call @g(%x) : (tuple<i32>) -> i64
  return %r : i64
}
