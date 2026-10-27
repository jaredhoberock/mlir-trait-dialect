// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A derive and a projection stand in a region that enforces SSA dominance, so
// a claim either reads is defined before it and never computed from its own
// result. In a module body, a graph region, a derive may be given its own
// result, a derivation resting on itself; each is refused where it stands.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %foo: !trait.claim<@Foo[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
// expected-error @below {{'trait.derive' op must be in a region that enforces SSA dominance}}
%d = trait.derive @Foo[i32] from @Foo_all[i32] given(%d) : (!trait.claim<@Foo[i32]>)

// -----

// The same derive with a call reading it.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %foo: !trait.claim<@Foo[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
// expected-error @below {{'trait.derive' op must be in a region that enforces SSA dominance}}
%d = trait.derive @Foo[i32] from @Foo_all[i32] given(%d) : (!trait.claim<@Foo[i32]>)
%r = trait.method.call %d @m() : !trait.claim<@Foo[i32]>, () -> i64

// -----

// Two projections each reading the other's result.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @B(%self: !trait.claim<@B[!T]>) -> !trait.claim<@A[!T]> {}
// expected-error @below {{'trait.project' op must be in a region that enforces SSA dominance}}
%a = trait.project %b[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
%b = trait.project %a[0] : !trait.claim<@A[i32]> -> !trait.claim<@B[i32]>
