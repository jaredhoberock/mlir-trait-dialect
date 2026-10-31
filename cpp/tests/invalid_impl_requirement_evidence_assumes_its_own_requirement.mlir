// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B requires its Out to be i64 and A[Out]. @B_bad binds Out to i32, alleges
// the false equality, and derives A[B[i8]::Out] from @A_i64. An impl's body
// computes its requirement evidence, which may not assume the impl's own
// application or the requirements it carries: the
// equality requirement is no hypothesis there, so the derive's claim does not
// read as @A[i64] and the derive is refused.

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!T]>, !T) -> i64
trait.trait private @B(!trait.claim<@B[!T]>) -> (!trait.claim<!trait.proj<@B[!T], "Out"> = i64>, !trait.claim<@A[!trait.proj<@B[!T], "Out">]>) assoc {Out = 0}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) members [@A.a -> @A_i64.a] {
}
func.func private @A_i64.a(%self: !trait.claim<@A[i64]>, %x: i64) -> i64 {
  %c = arith.constant 64 : i64
  return %c : i64
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>, %x: i32) -> i64 {
  %c = arith.constant 32 : i64
  return %c : i64
}
trait.impl private @B_bad(%self: !trait.claim<@B[i8]>) bindings {Out = i32} {
  %e = trait.allege !trait.proj<@B[i8], "Out"> = i64
  // expected-error @below {{impl '@A_i64' at the arguments the citation gives it proves '!trait.claim<@A[i64]>', not '!trait.claim<@A[!trait.proj<@B[i8], "Out">]>'}}
  %d = trait.derive @A[!trait.proj<@B[i8], "Out">] from @A_i64 given()
  trait.return %e, %d : !trait.claim<!trait.proj<@B[i8], "Out"> = i64>, !trait.claim<@A[!trait.proj<@B[i8], "Out">]>
}
