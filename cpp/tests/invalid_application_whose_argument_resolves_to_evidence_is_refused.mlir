// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An impl may bind an associated type to a type carrying evidence -- the tuple
// dialect's mappers bind their claims so -- but a predicate states types, never
// evidence, so an application whose argument resolves to such a type is about
// no type, though its arguments as written name no claim. Selection reads the
// application before it selects an impl for it and refuses it there, before
// any proof over it or equality respelling it is written, naming the type
// carrying the claim that the predicate it reads resolves to: an allegation, a
// projection whose head is such an application, a method call's receiver, a
// proof cited by hand, which erasure's entry reads as selection does, and a
// parameter's claim, which instantiation names where it stands unproven.

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_any(%self: !trait.claim<@B[!trait.poly<0>]>) {}
// expected-error @below {{unproven monomorphic claim '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>' after instantiate-monomorphs}}
func.func private @need(!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>)
func.func @run() {
  // expected-error @below {{'trait.allege' op a predicate states types, and '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>' resolves to 'tuple<!trait.claim<@Mark[i32]>>', which carries a claim}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>' after instantiate-monomorphs}}
  %c = trait.allege @B[!trait.proj<@Wrap[i32], "Item">]
  func.call @need(%c) : (!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>) -> ()
  return
}

// -----

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }
trait.trait private @Gat(%self: !trait.claim<@Gat[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Gat_any(%self: !trait.claim<@Gat[!trait.poly<0>]>) { trait.assoc_type @Item = i64 }
func.func @run() {
  // expected-error @below {{'trait.allege' op a predicate states types, and '!trait.claim<!trait.proj<@Gat[!trait.proj<@Wrap[i32], "Item">], "Item"> = i64>' resolves to '!trait.proj<@Gat[tuple<!trait.claim<@Mark[i32]>>], "Item">', which carries a claim}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Gat[!trait.proj<@Wrap[i32], "Item">], "Item"> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Gat[!trait.proj<@Wrap[i32], "Item">], "Item"> = i64
  return
}

// -----

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }
trait.trait private @Show(%self: !trait.claim<@Show[!trait.poly<0>]>) { trait.method @v() -> i64 }
trait.impl private @Show_any(%self: !trait.claim<@Show[!trait.poly<0>]>) {
  trait.method @v() -> i64 { %k = arith.constant 5 : i64  trait.return %k : i64 }
}
func.func @run() -> i64 {
  // expected-error @below {{'trait.allege' op a predicate states types, and '!trait.claim<@Show[!trait.proj<@Wrap[i32], "Item">]>' resolves to 'tuple<!trait.claim<@Mark[i32]>>', which carries a claim}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@Show[!trait.proj<@Wrap[i32], "Item">]>' after instantiate-monomorphs}}
  %c = trait.allege @Show[!trait.proj<@Wrap[i32], "Item">]
  %v = trait.method.call %c @v() : !trait.claim<@Show[!trait.proj<@Wrap[i32], "Item">]>, () -> i64
  return %v : i64
}

// -----

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_any(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.proof private @p {
  %d = trait.derive @B[!trait.proj<@Wrap[i32], "Item">] from @B_any[!trait.proj<@Wrap[i32], "Item">] given()
  trait.return %d : !trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>
}
// expected-error @below {{a predicate states types, and '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">] by @p>' resolves to 'tuple<!trait.claim<@Mark[i32]>>', which carries a claim}}
// expected-error @below {{claim '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">] by @p>' is not proven before erasure: an argument it states resolves to no type a predicate states}}
func.func @use(%w: !trait.claim<@B[!trait.proj<@Wrap[i32], "Item">] by @p>) -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}
func.func @run() -> i64 {
  // expected-error @below {{a predicate states types, and '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">] by @p>' resolves to 'tuple<!trait.claim<@Mark[i32]>>', which carries a claim}}
  // expected-error @below {{claim '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">] by @p>' is not proven before erasure: an argument it states resolves to no type a predicate states}}
  %w = trait.witness @p for @B[!trait.proj<@Wrap[i32], "Item">]
  %v = func.call @use(%w) : (!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">] by @p>) -> i64
  return %v : i64
}

// -----

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_any(%self: !trait.claim<@B[!trait.poly<0>]>) {}
// expected-error @below {{a predicate states types, and '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>' resolves to 'tuple<!trait.claim<@Mark[i32]>>', which carries a claim}}
// expected-error @below {{unproven monomorphic claim '!trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>' after instantiate-monomorphs}}
func.func @g(%c: !trait.claim<@B[!trait.proj<@Wrap[i32], "Item">]>) -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}
