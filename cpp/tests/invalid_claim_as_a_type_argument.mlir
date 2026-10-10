// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A proof lives at the root of a value's claim type, so a claim is never a type
// argument. A caller binding a generic function's parameter to a proven claim,
// where the function derives through a premise over that parameter, spells the
// claim inside a trait application; that application is refused where it is
// written, before any pass reads it.

trait.trait private @M(%s: !trait.claim<@M[!trait.poly<0>]>) {}
trait.impl private @M_i32(%s: !trait.claim<@M[i32]>) {}
trait.trait private @B(%s: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_all(%s: !trait.claim<@B[!trait.poly<0>]>) {}
trait.proof private @pb {
  %d = trait.derive @B[!trait.claim<@M[i32] by @M_i32>] from @B_all[!trait.claim<@M[i32] by @M_i32>] given()
  trait.return %d : !trait.claim<@B[!trait.claim<@M[i32] by @M_i32>]>
}
trait.trait private @A(%s: !trait.claim<@A[!trait.poly<0>]>) {}
trait.impl private @A_all(%s: !trait.claim<@A[!trait.poly<0>]>, %b: !trait.claim<@B[!trait.poly<0>]>) {}
func.func private @g(%b: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  %d = trait.derive @A[!trait.poly<0>] from @A_all[!trait.poly<0>] given(%b) : (!trait.claim<@B[!trait.poly<0>]>)
  return %d : !trait.claim<@A[!trait.poly<0>]>
}
func.func @test() {
  // expected-error @below {{trait application #trait<application@B[!trait.claim<@M[i32] by @M_i32>]> takes a claim as a type argument}}
  %b = trait.witness @pb for @B[!trait.claim<@M[i32] by @M_i32>]
  %a = trait.func.call @g(%b) : (!trait.claim<@B[!trait.claim<@M[i32] by @M_i32>] by @pb>) -> !trait.claim<@A[!trait.claim<@M[i32] by @M_i32>]>
  return
}

// -----

// A projection's own arguments are type arguments too.

trait.trait private @M(%s: !trait.claim<@M[!trait.poly<0>]>) {}
trait.trait private @G(%s: !trait.claim<@G[!trait.poly<0>]>) {
  trait.assoc_type @Out<[!trait.poly<1>]>
}
// expected-error @below {{projection '!trait.proj<@G[i32], "Out", [!trait.claim<@M[i32]>]>' takes a claim as a type argument}}
func.func private @f(%x: !trait.proj<@G[i32], "Out", [!trait.claim<@M[i32]>]>)
