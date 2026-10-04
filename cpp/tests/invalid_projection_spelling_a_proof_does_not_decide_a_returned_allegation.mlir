// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A proof a projection spells is checked against the evidence it reads, never
// taken as the decision. @Host_i64 returns an allegation of @Extra[i64], and
// two impls bind @Extra[i64]; the projection spelled by @Extra7 is replaced by
// that allegation, which selection decides where it stands and refuses as
// ambiguous, as it refuses the unspelled projection.

// CHECK: :[[@LINE+12]]:{{[0-9]+}}: error: 'trait.allege' op incoherent impls (multiple satisfiable) for '!trait.claim<@Extra[i64]>'

!T = !trait.poly<0>
trait.trait private @Extra(%self: !trait.claim<@Extra[!T]>) { trait.method @v() -> i64 }
trait.impl private @Extra7(%self: !trait.claim<@Extra[i64]>) {
  trait.method @v() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
trait.impl private @Extra9(%self: !trait.claim<@Extra[i64]>) {
  trait.method @v() -> i64 { %c = arith.constant 9 : i64  trait.return %c : i64 }
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) -> !trait.claim<@Extra[!T]> {}
trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) {
  %e = trait.allege @Extra[i64]
  trait.return %e : !trait.claim<@Extra[i64]>
}
func.func @main() -> i64 {
  %h = trait.witness @Host_i64 for @Host[i64]
  %e = trait.project %h[0] : !trait.claim<@Host[i64] by @Host_i64> -> !trait.claim<@Extra[i64] by @Extra7>
  %v = trait.method.call %e @Extra[i64]::@v() : () -> i64 by @Extra7
  return %v : i64
}

// -----

// The same with the allegation given to a derive the return computes: the
// projection spelled by @P7, a proof of that derive over @Extra7, is replaced
// by the derive, whose premise selection decides and refuses as ambiguous.

// CHECK: :[[@LINE+24]]:{{[0-9]+}}: error: 'trait.allege' op incoherent impls (multiple satisfiable) for '!trait.claim<@Extra[i64]>'

!T = !trait.poly<0>
trait.trait private @Extra(%self: !trait.claim<@Extra[!T]>) { trait.method @v() -> i64 }
trait.impl private @Extra7(%self: !trait.claim<@Extra[i64]>) {
  trait.method @v() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
trait.impl private @Extra9(%self: !trait.claim<@Extra[i64]>) {
  trait.method @v() -> i64 { %c = arith.constant 9 : i64  trait.return %c : i64 }
}
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Nine(%self: !trait.claim<@Mark[i64]>, %e: !trait.claim<@Extra[i64]>) {
  trait.method @value() -> i64 {
    %v = trait.method.call %e @Extra[i64]::@v() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @P7 {
  %e = trait.witness @Extra7 for @Extra[i64]
  %d = trait.derive @Mark[i64] from @Nine given(%e) : (!trait.claim<@Extra[i64] by @Extra7>)
  trait.return %d : !trait.claim<@Mark[i64]>
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) -> !trait.claim<@Mark[!T]> {}
trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) {
  %e = trait.allege @Extra[i64]
  %m = trait.derive @Mark[i64] from @Nine given(%e) : (!trait.claim<@Extra[i64]>)
  trait.return %m : !trait.claim<@Mark[i64]>
}
func.func @main() -> i64 {
  %h = trait.witness @Host_i64 for @Host[i64]
  %m = trait.project %h[0] : !trait.claim<@Host[i64] by @Host_i64> -> !trait.claim<@Mark[i64] by @P7>
  %v = trait.method.call %m @Mark[i64]::@value() : () -> i64 by @P7
  return %v : i64
}
