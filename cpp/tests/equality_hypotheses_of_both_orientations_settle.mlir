// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An equality hypothesis relates its two types; it does not rewrite the one
// into the other. @outer's where clause holds @Trait[!T]::Item = @Other[!T]::Item
// and @inner's holds the same equality written the other way round, so the call
// stands under both orientations at once: read as directed rules they carry the
// spelling from one projection to the other and back for as long as anything
// asks. Read as one class of two members, every member rewrites to the member
// the class stands for and the call's signature check settles at once.

// CHECK-LABEL: func.func @outer
// CHECK: trait.func.call @inner

!T = !trait.poly<0>
trait.trait private @Trait[!T] {
  trait.assoc_type @Item
}

!U = !trait.poly<1>
trait.trait private @Other[!U] {
  trait.assoc_type @Item
}

!A = !trait.poly<2>
func.func private @inner(!A, !trait.claim<@Trait[!A]>, !trait.claim<@Other[!A]>,
    !trait.claim<!trait.proj<@Other[!A], "Item"> = !trait.proj<@Trait[!A], "Item">>)
    -> !trait.proj<@Other[!A], "Item">

!B = !trait.poly<3>
func.func @outer(%x: !B, %t: !trait.claim<@Trait[!B]>, %o: !trait.claim<@Other[!B]>,
    %eq: !trait.claim<!trait.proj<@Trait[!B], "Item"> = !trait.proj<@Other[!B], "Item">>)
    -> !trait.proj<@Other[!B], "Item"> {
  %rev = trait.witness compose(%eq)
    : (!trait.claim<!trait.proj<@Trait[!B], "Item"> = !trait.proj<@Other[!B], "Item">>)
    : !trait.claim<!trait.proj<@Other[!B], "Item"> = !trait.proj<@Trait[!B], "Item">>
  %r = trait.func.call @inner(%x, %t, %o, %rev)
    : (!B, !trait.claim<@Trait[!B]>, !trait.claim<@Other[!B]>,
       !trait.claim<!trait.proj<@Other[!B], "Item"> = !trait.proj<@Trait[!B], "Item">>)
    -> !trait.proj<@Other[!B], "Item">
  return %r : !trait.proj<@Other[!B], "Item">
}
