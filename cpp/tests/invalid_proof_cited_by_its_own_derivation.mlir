// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics | FileCheck %s

// A proof's derivation is its derive and the claims the derive is given,
// computed in the proof's body, and the proofs the witnesses there name are
// the proofs it cites. A proof whose derivation cites it, itself or through
// the proofs it cites, rests on nothing: impl selection refuses the same
// application as a cycle, so the stage refuses the proof, naming the cycle.
// @p derives @Foo[i32] from @Foo_all, whose where entry @Foo[i32] it supplies
// with its own witness.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %foo: !trait.claim<@Foo[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
// expected-error @below {{'trait.proof' op is cited by its own derivation: @p -> @p}}
trait.proof private @p {
  %w = trait.witness @p for @Foo[i32]
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%w) : (!trait.claim<@Foo[i32] by @p>)
  trait.return %d : !trait.claim<@Foo[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @p for @Foo[i32]
  %r = trait.method.call %w @Foo[i32]::@m() : () -> i64 by @p
  return %r : i64
}

// -----

// Two proofs each supplying the other's where entry.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) {}
trait.trait private @Baz(%self: !trait.claim<@Baz[!T]>) {}
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %b: !trait.claim<@Baz[!T]>) {}
trait.impl private @Baz_all(%self: !trait.claim<@Baz[!T]>, %f: !trait.claim<@Foo[!T]>) {}
// expected-error @below {{'trait.proof' op is cited by its own derivation: @p -> @q -> @p}}
trait.proof private @p {
  %b = trait.witness @q for @Baz[i32]
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%b) : (!trait.claim<@Baz[i32] by @q>)
  trait.return %d : !trait.claim<@Foo[i32]>
}
trait.proof private @q {
  %f = trait.witness @p for @Foo[i32]
  %d = trait.derive @Baz[i32] from @Baz_all[i32] given(%f) : (!trait.claim<@Foo[i32] by @p>)
  trait.return %d : !trait.claim<@Baz[i32]>
}

// -----

// Through an equality where entry: @Foo_all needs @Bar[i32]::Out = i64,
// which @Bar_all binds given @Foo[i32], and @p gives that the witness of
// itself.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) {}
trait.trait private @Bar(%self: !trait.claim<@Bar[!T]>) { trait.assoc_type @Out }
trait.impl private @Bar_all(%self: !trait.claim<@Bar[!T]>, %f: !trait.claim<@Foo[!T]>) { trait.assoc_type @Out = i64 }
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %e: !trait.claim<!trait.proj<@Bar[!T], "Out"> = i64>) {}
// expected-error @below {{'trait.proof' op is cited by its own derivation: @p -> @p}}
trait.proof private @p {
  %w = trait.witness @p for @Foo[i32]
  %e = trait.witness proj_resolve !trait.proj<@Bar[i32], "Out"> resolves i64 by @Bar_all[i32] given(%w)
    : (!trait.claim<@Foo[i32] by @p>) : !trait.claim<!trait.proj<@Bar[i32], "Out"> = i64>
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%e) : (!trait.claim<!trait.proj<@Bar[i32], "Out"> = i64>)
  trait.return %d : !trait.claim<@Foo[i32]>
}

// -----

// Through the derive of a where entry, computed in the proof's body.

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) {}
trait.trait private @Baz(%self: !trait.claim<@Baz[!T]>) {}
trait.impl private @Baz_all(%self: !trait.claim<@Baz[!T]>, %f: !trait.claim<@Foo[!T]>) {}
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %b: !trait.claim<@Baz[!T]>) {}
// expected-error @below {{'trait.proof' op is cited by its own derivation: @p -> @p}}
trait.proof private @p {
  %w = trait.witness @p for @Foo[i32]
  %b = trait.derive @Baz[i32] from @Baz_all[i32] given(%w) : (!trait.claim<@Foo[i32] by @p>)
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%b) : (!trait.claim<@Baz[i32]>)
  trait.return %d : !trait.claim<@Foo[i32]>
}

// -----

// A proof citing another proof, which cites none, rests on that one: @p's
// where entry is @q's, and the call runs @Foo_all's method.

// CHECK-LABEL: func.func @main() -> i64
// CHECK:         call @Foo_all_{{h[0-9a-f]+}}_m()

!T = !trait.poly<0>
trait.trait private @Foo(%self: !trait.claim<@Foo[!T]>) { trait.method @m() -> i64 }
trait.trait private @Baz(%self: !trait.claim<@Baz[!T]>) {}
trait.impl private @Baz_i32(%self: !trait.claim<@Baz[i32]>) {}
trait.impl private @Foo_all(%self: !trait.claim<@Foo[!T]>, %b: !trait.claim<@Baz[!T]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
trait.proof private @q {
  %d = trait.derive @Baz[i32] from @Baz_i32 given()
  trait.return %d : !trait.claim<@Baz[i32]>
}
trait.proof private @p {
  %b = trait.witness @q for @Baz[i32]
  %d = trait.derive @Foo[i32] from @Foo_all[i32] given(%b) : (!trait.claim<@Baz[i32] by @q>)
  trait.return %d : !trait.claim<@Foo[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @p for @Foo[i32]
  %r = trait.method.call %w @Foo[i32]::@m() : () -> i64 by @p
  return %r : i64
}
