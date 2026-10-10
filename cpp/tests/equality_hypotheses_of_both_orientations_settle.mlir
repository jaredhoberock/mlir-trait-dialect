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
trait.trait private @Trait(%self: !trait.claim<@Trait[!T]>) {
  trait.assoc_type @Item
}

!U = !trait.poly<1>
trait.trait private @Other(%self: !trait.claim<@Other[!trait.poly<0>]>) {
  trait.assoc_type @Item
}

!A = !trait.poly<2>
func.func private @inner(!trait.poly<0>, !trait.claim<@Trait[!trait.poly<0>]>, !trait.claim<@Other[!trait.poly<0>]>,
    !trait.claim<!trait.proj<@Other[!trait.poly<0>], "Item"> = !trait.proj<@Trait[!trait.poly<0>], "Item">>)
    -> !trait.proj<@Other[!trait.poly<0>], "Item">

!B = !trait.poly<3>
func.func @outer(%x: !trait.poly<0>, %t: !trait.claim<@Trait[!trait.poly<0>]>, %o: !trait.claim<@Other[!trait.poly<0>]>,
    %eq: !trait.claim<!trait.proj<@Trait[!trait.poly<0>], "Item"> = !trait.proj<@Other[!trait.poly<0>], "Item">>)
    -> !trait.proj<@Other[!trait.poly<0>], "Item"> {
  %rev = trait.witness compose(%eq)
    : (!trait.claim<!trait.proj<@Trait[!trait.poly<0>], "Item"> = !trait.proj<@Other[!trait.poly<0>], "Item">>)
    : !trait.claim<!trait.proj<@Other[!trait.poly<0>], "Item"> = !trait.proj<@Trait[!trait.poly<0>], "Item">>
  %r = trait.func.call @inner(%x, %t, %o, %rev)
    : (!trait.poly<0>, !trait.claim<@Trait[!trait.poly<0>]>, !trait.claim<@Other[!trait.poly<0>]>,
       !trait.claim<!trait.proj<@Other[!trait.poly<0>], "Item"> = !trait.proj<@Trait[!trait.poly<0>], "Item">>)
    -> !trait.proj<@Other[!trait.poly<0>], "Item">
  return %r : !trait.proj<@Other[!trait.poly<0>], "Item">
}
