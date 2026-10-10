// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A derive is proven by the impl it names, over the operands it is given: at
// i32 @g's derive names @Tr_any, whose premise @Mark[i32] no impl proves, so
// the allegation it is given is refused, and so is the derive standing on it,
// although selection could prove @Tr[tuple<i32>] through @Tr_i32 instead. The
// derive is never answered by selecting again.

// CHECK: error: unproven monomorphic claim '!trait.claim<@Mark[i32]>' after instantiate-monomorphs
// CHECK: error: unproven monomorphic claim '!trait.claim<@Tr[tuple<i32>]>' after instantiate-monomorphs
// CHECK-NOT: error:

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) {}
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) {
  trait.method @get(!T) -> i64
}

trait.impl private @Tr_any(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>, %mark: !trait.claim<@Mark[!trait.poly<0>]>) {
  trait.method @get(%x: tuple<!trait.poly<0>>) -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

trait.impl private @Tr_i32(%self: !trait.claim<@Tr[tuple<i32>]>) {
  trait.method @get(%x: tuple<i32>) -> i64 {
    %c = arith.constant 2 : i64
    trait.return %c : i64
  }
}

func.func private @g(%x: tuple<!T>) -> i64 {
  %m = trait.allege @Mark[!T]
  %d = trait.derive @Tr[tuple<!T>] from @Tr_any given(%m) : (!trait.claim<@Mark[!T]>)
  %r = trait.method.call %d @Tr[tuple<!T>]::@get(%x) : (tuple<!T>) -> i64
  return %r : i64
}

func.func @main(%x: tuple<i32>) -> i64 {
  %r = trait.func.call @g(%x) : (tuple<i32>) -> i64
  return %r : i64
}
