// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics | FileCheck %s

// @I binds @A[i8]::Out to i32 and returns an allegation that it is i64 for the
// trait's requirement. The projection reading that requirement is replaced by
// that allegation, which is decided there and refused, even where the only use
// of the projection is a coerce whose input and result are already one type:
// the coerce settles at once, and the projection, an obligation outstanding,
// stands to be inlined and decided rather than being erased unread.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i32
  // expected-error @below {{alleges '!trait.proj<@A[i8], "Out">' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@A[i8], "Out"> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i8]
  %e = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
  return %r : i64
}

// -----

// The same program with @I binding @A[i8]::Out to i64: the allegation holds,
// and the coerce, once decided, is its input.

// CHECK-LABEL: func.func @main() -> i64
// CHECK-NEXT:    %[[NINE:.*]] = arith.constant 9 : i64
// CHECK-NEXT:    return %[[NINE]] : i64

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i64
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i8]
  %e = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
  return %r : i64
}
