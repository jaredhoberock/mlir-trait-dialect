// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// An op reading evidence reads it where SSA dominance holds in every region
// up to its nearest isolated ancestor: a region nested in a module body may
// read a value the module body defines after it, so a derive standing in
// such a region could be given its own result. Each op below reads a value
// computed from its own result, and is refused where it stands.

// A derive in a region nested in the module body, given the value the region
// yields.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %foo: !trait.claim<@Foo[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
%x = scf.execute_region -> !trait.claim<@Foo[i32]> {
  // expected-error @below {{'trait.derive' op must be in a region that enforces SSA dominance}}
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%x) : (!trait.claim<@Foo[i32]>)
  scf.yield %d : !trait.claim<@Foo[i32]>
}

// -----

// The same through a second region in the module body.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %foo: !trait.claim<@Foo[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
%x = scf.execute_region -> !trait.claim<@Foo[i32]> {
  // expected-error @below {{'trait.derive' op must be in a region that enforces SSA dominance}}
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%y) : (!trait.claim<@Foo[i32]>)
  scf.yield %d : !trait.claim<@Foo[i32]>
}
%y = scf.execute_region -> !trait.claim<@Foo[i32]> {
  scf.yield %x : !trait.claim<@Foo[i32]>
}

// -----

// The first with a call reading the region's value.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %foo: !trait.claim<@Foo[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
%x = scf.execute_region -> !trait.claim<@Foo[i32]> {
  // expected-error @below {{'trait.derive' op must be in a region that enforces SSA dominance}}
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%x) : (!trait.claim<@Foo[i32]>)
  scf.yield %d : !trait.claim<@Foo[i32]>
}
%r = trait.method.call %x @m() : !trait.claim<@Foo[i32]>, () -> i64

// -----

// A projection-resolution witness in the module body given its own result for
// its impl's equality where entry.

!T = !trait.poly<0>
trait.trait private @Bar(%self: !trait.claim<@Bar[!T]>) { trait.assoc_type @Out }
trait.impl private @Bar_all(%self: !trait.claim<@Bar[!T]>, %p: !trait.claim<!trait.proj<@Bar[!T], "Out"> = i64>) { trait.assoc_type @Out = i64 }
// expected-error @below {{'trait.witness' op must be in a region that enforces SSA dominance}}
%e = trait.witness proj_resolve !trait.proj<@Bar[i32], "Out"> resolves i64 by @Bar_all[i32] given(%e) : (!trait.claim<!trait.proj<@Bar[i32], "Out"> = i64>) : !trait.claim<!trait.proj<@Bar[i32], "Out"> = i64>

// -----

// Two claim-returning requirement calls in the module body, each the other's
// receiver.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
  trait.method @requirement_0() -> !trait.claim<@B[!T]>
}
trait.trait private @B(%self: !trait.claim<@B[!T]>) {
  trait.method @requirement_0() -> !trait.claim<@A[!T]>
}
// expected-error @below {{'trait.method.call' op must be in a region that enforces SSA dominance}}
%b = trait.method.call %a @requirement_0() : !trait.claim<@A[i32]>, () -> !trait.claim<@B[i32]>
%a = trait.method.call %b @requirement_0() : !trait.claim<@B[i32]>, () -> !trait.claim<@A[i32]>
