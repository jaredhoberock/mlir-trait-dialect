// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A call of a quantified requirement's method computes evidence: it is
// replaced by the body its receiver's impl defines, so the evidence is the
// operand that body returns, and the call's claim names no proof, as a
// projection of a trait requirement names none. Each call below names @Seven's
// evidence where the body returns @Nine's, and is refused where it stands.
// @W's @requirement_0 returns its parameter, given @Nine's witness.

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 { %v = arith.constant 7 : i64  trait.return %v : i64 }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 { %v = arith.constant 9 : i64  trait.return %v : i64 }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) {
  trait.method @requirement_0(!trait.claim<@Mark[!T]>) -> !trait.claim<@Mark[!T]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  trait.method @requirement_0(%p: !trait.claim<@Mark[i32]>) -> !trait.claim<@Mark[i32]> {
    trait.return %p : !trait.claim<@Mark[i32]>
  }
}
func.func @main() -> i64 {
  %w = trait.witness @W for @Wrapped[i32]
  %p = trait.witness @Nine for @Mark[i32]
  // expected-error @below {{'trait.method.call' op computes the evidence its receiver's impl returns, so its claim names no proof, found '!trait.claim<@Mark[i32] by @Seven>'}}
  %m = trait.method.call %w @requirement_0(%p) : !trait.claim<@Wrapped[i32] by @W>, (!trait.claim<@Mark[i32] by @Nine>) -> !trait.claim<@Mark[i32] by @Seven>
  %v = trait.method.call %m @value() : !trait.claim<@Mark[i32] by @Seven>, () -> i64
  return %v : i64
}

// -----

// @W's @requirement_0 returns @Nine's witness respelled at its associated
// type; the call names @Seven_as, a proof of the same respelling cast from
// @Seven.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 { %v = arith.constant 7 : i64  trait.return %v : i64 }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 { %v = arith.constant 9 : i64  trait.return %v : i64 }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Wrapped[!T], "Item", [!U]>]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = i32
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [!trait.poly<0>]>]> {
    %n = trait.witness @Nine for @Mark[i32]
    %e = trait.witness proj_resolve !trait.proj<@Wrapped[i32], "Item", [!trait.poly<0>]> resolves i32 by @W : !trait.claim<!trait.proj<@Wrapped[i32], "Item", [!trait.poly<0>]> = i32>
    %m = trait.coerce %n : !trait.claim<@Mark[i32] by @Nine> to !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [!trait.poly<0>]>]> via (%e) : (!trait.claim<!trait.proj<@Wrapped[i32], "Item", [!trait.poly<0>]> = i32>)
    trait.return %m : !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [!trait.poly<0>]>]>
  }
}
trait.proof private @Seven_as {
  %s = trait.witness @Seven for @Mark[i32]
  %e = trait.witness proj_resolve !trait.proj<@Wrapped[i32], "Item", [i1]> resolves i32 by @W : !trait.claim<!trait.proj<@Wrapped[i32], "Item", [i1]> = i32>
  %c = trait.coerce %s : !trait.claim<@Mark[i32] by @Seven> to !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>]> via (%e) : (!trait.claim<!trait.proj<@Wrapped[i32], "Item", [i1]> = i32>)
  trait.return %c : !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>]>
}
func.func @main() -> i64 {
  %w = trait.witness @W for @Wrapped[i32]
  // expected-error @below {{'trait.method.call' op computes the evidence its receiver's impl returns, so its claim names no proof}}
  %m = trait.method.call %w @requirement_0() : !trait.claim<@Wrapped[i32] by @W>, () -> !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>] by @Seven_as>
  %v = trait.method.call %m @value() : !trait.claim<@Mark[!trait.proj<@Wrapped[i32], "Item", [i1]>] by @Seven_as>, () -> i64
  return %v : i64
}

// -----

// @W's @requirement_0 returns a derive of @Mark[i32] from @Nine; the call
// names @Seven, and a coercion dropping that proof reads it.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 { %c = arith.constant 9 : i64  trait.return %c : i64 }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) {
  trait.method @requirement_0(!U) -> !trait.claim<@Mark[i32]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  trait.method @requirement_0(%u: !trait.poly<0>) -> !trait.claim<@Mark[i32]> {
    %n = trait.derive @Mark[i32] from @Nine given()
    trait.return %n : !trait.claim<@Mark[i32]>
  }
}
func.func @main() -> i64 {
  %w = trait.witness @W for @Wrapped[i32]
  %u = arith.constant 0 : i1
  // expected-error @below {{'trait.method.call' op computes the evidence its receiver's impl returns, so its claim names no proof, found '!trait.claim<@Mark[i32] by @Seven>'}}
  %m = trait.method.call %w @requirement_0(%u) : !trait.claim<@Wrapped[i32] by @W>, (i1) -> !trait.claim<@Mark[i32] by @Seven>
  %c = trait.coerce %m : !trait.claim<@Mark[i32] by @Seven> to !trait.claim<@Mark[i32]>
  %v = trait.method.call %c @value() : !trait.claim<@Mark[i32]>, () -> i64
  return %v : i64
}
