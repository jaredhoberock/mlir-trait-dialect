// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A method's body ends in trait.return, never func.return.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @a(!T) -> i64 }
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @a(%x: i64) -> i64 {
    // expected-error @below {{'func.return' op expects parent op 'func.func'}}
    func.return %x : i64
  }
}

// -----

// trait.return ends a method's, an impl's or a proof's body and nothing else.

func.func @f(%x: i64) -> i64 {
  // expected-error @below {{'trait.return' op expects parent op to be one of 'trait.method, trait.impl, trait.proof'}}
  trait.return %x : i64
}

// -----

// A return's operands are the method's results: as many as it returns...

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @a(!T) -> i64 }
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @a(%x: i64) -> i64 {
    // expected-error @below {{'trait.return' op has 2 operands, but enclosing method (@a) returns 1}}
    trait.return %x, %x : i64, i64
  }
}

// -----

// ...each of the type it returns.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @a(!T) -> i64 }
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @a(%x: i64) -> i64 {
    %y = arith.trunci %x : i64 to i32
    // expected-error @below {{'trait.return' op type of return operand 0 ('i32') doesn't match method result type ('i64') in method @a}}
    trait.return %y : i32
  }
}

// -----

// A method lives and dies with its trait or impl, so it carries no visibility.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{'trait.method' op must carry no visibility: a method lives and dies with its trait or impl}}
  trait.method private @a(!T) -> i64
}

// -----

// A method is a member of a trait or an impl.

// expected-error @below {{'trait.method' op expects parent op to be one of 'trait.trait, trait.impl'}}
trait.method @a(%x: i64) -> i64 {
  trait.return %x : i64
}

// -----

// A trait's or impl's body holds methods and associated types and no value: a
// constant written there is refused, as one the folder pooled there would be.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @a(!T) -> i64 }
// expected-error @below {{'trait.impl' op unexpected child op 'arith.constant'}}
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  %seven = arith.constant 7 : i64
  trait.method @a(%x: i64) -> i64 {
    trait.return %x : i64
  }
}

// -----

// An impl's block arguments are claims -- its own application, then one per
// where entry -- so a method reads nothing but evidence from outside its own
// body.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @a(!T) -> i64 }
// expected-error @below {{'trait.impl' op where entry 0 must be an unproven claim, found 'i64'}}
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>, %seven: i64) {
  trait.method @a(%x: i64) -> i64 {
    trait.return %seven : i64
  }
}

// -----

// An impl's method has a body: a method without one is a trait's requirement.

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @a(!T) -> i64 }
// expected-error @below {{'trait.impl' op method 'a' must have body}}
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @a(i64) -> i64
}
