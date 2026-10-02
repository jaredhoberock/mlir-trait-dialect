// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B requires its Out to be i64 and A[Out]. @B_bad binds Out to i32, alleges
// the false equality, and derives A[B[i8]::Out] from @A_i64. Outside its
// methods an impl's body computes its requirement evidence, which may not
// assume the impl's own application or the requirements it carries: the
// equality requirement is no hypothesis there, so the derive's claim does not
// read as @A[i64] and the derive is refused.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) { trait.method @a(!T) -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!T]>) -> (!trait.claim<!trait.proj<@B[!T], "Out"> = i64>, !trait.claim<@A[!trait.proj<@B[!T], "Out">]>) {
  trait.assoc_type @Out
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
  trait.method @a(%x: i64) -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @a(%x: i32) -> i64 {
    %c = arith.constant 32 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_bad(%self: !trait.claim<@B[i8]>) {
  trait.assoc_type @Out = i32
  %e = trait.allege !trait.proj<@B[i8], "Out"> = i64
  // expected-error @below {{impl '@A_i64' at the arguments the citation gives it proves '!trait.claim<@A[i64]>', not '!trait.claim<@A[!trait.proj<@B[i8], "Out">]>'}}
  %d = trait.derive @A[!trait.proj<@B[i8], "Out">] from @A_i64 given()
  trait.return %e, %d : !trait.claim<!trait.proj<@B[i8], "Out"> = i64>, !trait.claim<@A[!trait.proj<@B[i8], "Out">]>
}
func.func @main() -> i64 {
  %b = trait.witness @B_bad for @B[i8]
  %a = trait.project %b[1] : !trait.claim<@B[i8] by @B_bad> -> !trait.claim<@A[i32]>
  %x = arith.constant 0 : i32
  %v = trait.method.call %a @A[i32]::@a(%x) : (i32) -> i64
  return %v : i64
}
