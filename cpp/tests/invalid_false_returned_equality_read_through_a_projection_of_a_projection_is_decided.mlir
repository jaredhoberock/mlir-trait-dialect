// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @J returns a witness of @I for @B's requirement @A[i8], and @I returns an
// allegation that @A[i8]::Out, bound to i32, is i64. Main projects @J's
// requirement, spelled without @I's proof, then the equality off that, and
// cites it in an identity coerce. The outer projection is replaced by @I's
// witness coerced to that unproven spelling, which the stage has yet to carry
// @I's proof through when the coerce is first considered; the inner projection
// is an obligation outstanding however its source is spelled, so it stands
// once the coerce settles, is replaced by the allegation, and the allegation
// is refused.
// The true control is
// equality_read_through_a_projection_of_a_projection_holds.
// CHECK: :[[@LINE+10]]:{{[0-9]+}}: error: 'trait.allege' op alleges '!trait.proj<@A[i8], "Out">' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
!E = !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.trait private @B(%self: !trait.claim<@B[!T]>) -> !trait.claim<@A[!T]> {}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i32
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !E
}
trait.impl private @J(%self: !trait.claim<@B[i8]>) {
  %w = trait.witness @I for @A[i8]
  trait.return %w : !trait.claim<@A[i8] by @I>
}
func.func @main() -> i64 {
  %b = trait.witness @J for @B[i8]
  %a = trait.project %b[0] : !trait.claim<@B[i8] by @J> -> !trait.claim<@A[i8]>
  %e = trait.project %a[0] : !trait.claim<@A[i8]> -> !E
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!E)
  return %r : i64
}

// -----

// The same with no associated type: @IB alleges that i32 is i64 (Sol's
// reduction; Astra reduced the same shape independently).
// CHECK: :[[@LINE+5]]:{{[0-9]+}}: error: 'trait.allege' op alleges 'i32' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @B(%s: !trait.claim<@B[!T]>) -> !trait.claim<!T = i64> {}
trait.impl private @IB(%s: !trait.claim<@B[i32]>) {
 %e = trait.allege i32 = i64
 trait.return %e : !trait.claim<i32 = i64>
}
trait.trait private @A(%s: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @IA(%s: !trait.claim<@A[i32]>) {
 %b = trait.witness @IB for @B[i32]
 trait.return %b : !trait.claim<@B[i32] by @IB>
}
func.func @main() -> i64 {
 %a = trait.witness @IA for @A[i32]
 %b = trait.project %a[0] : !trait.claim<@A[i32] by @IA> -> !trait.claim<@B[i32]>
 %e = trait.project %b[0] : !trait.claim<@B[i32]> -> !trait.claim<i32 = i64>
 %n = arith.constant 9 : i64
 %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<i32 = i64>)
 return %r : i64
}

// -----

// The same with the proof dropped by a coercion the program writes: the
// projection reads a source spelled without the proof it stands on.
// CHECK: :[[@LINE+5]]:{{[0-9]+}}: error: 'trait.allege' op alleges 'i32' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) -> !trait.claim<!T = i64> {}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  %e = trait.allege i32 = i64
  trait.return %e : !trait.claim<i32 = i64>
}
func.func @main() -> i64 {
 %w = trait.witness @I for @A[i32]
 %a = trait.coerce %w : !trait.claim<@A[i32] by @I> to !trait.claim<@A[i32]>
 %e = trait.project %a[0] : !trait.claim<@A[i32]> -> !trait.claim<i32 = i64>
 %n = arith.constant 9 : i64
 %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<i32 = i64>)
 return %r : i64
}
