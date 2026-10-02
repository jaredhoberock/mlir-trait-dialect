// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @I binds @A[i8]::Out to i32 and returns an allegation that it is i64. The
// projections reading that requirement stand inside the arms of an scf.if
// whose result an identity coerce cites. The coerce waits for what the arms
// yield, so the projections are replaced by the allegation, which is decided
// there and refused, rather than erased with the coerce's only use. The true
// control is false_equality_yielded_from_a_region_is_decided_where_it_holds.
// CHECK: :[[@LINE+8]]:{{[0-9]+}}: error: 'trait.allege' op alleges '!trait.proj<@A[i8], "Out">' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i32
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i8]
  %c = arith.constant true
  %e = scf.if %c -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64> {
    %p = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
    scf.yield %p : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
  } else {
    %p = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
    scf.yield %p : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
  }
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
  return %r : i64
}

// -----

// The same with one projection yielded from an scf.execute_region.
// CHECK: :[[@LINE+8]]:{{[0-9]+}}: error: 'trait.allege' op alleges '!trait.proj<@A[i8], "Out">' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> {
  trait.assoc_type @Out
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
  trait.assoc_type @Out = i32
  %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
  trait.return %e : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i8]
  %e = scf.execute_region -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64> {
    %p = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
    scf.yield %p : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
  }
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
  return %r : i64
}

// -----

// The same with no associated type: @I returns an allegation that i32 is i64
// for a requirement its trait states over its argument.
// CHECK: :[[@LINE+5]]:{{[0-9]+}}: error: 'trait.allege' op alleges 'i32' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) -> !trait.claim<!T = i64> {}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  %e = trait.allege i32 = i64
  trait.return %e : !trait.claim<i32 = i64>
}
func.func @main() -> i64 {
  %a = trait.witness @I for @A[i32]
  %e = scf.execute_region -> !trait.claim<i32 = i64> {
    %p = trait.project %a[0] : !trait.claim<@A[i32] by @I> -> !trait.claim<i32 = i64>
    scf.yield %p : !trait.claim<i32 = i64>
  }
  %n = arith.constant 9 : i64
  %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<i32 = i64>)
  return %r : i64
}
