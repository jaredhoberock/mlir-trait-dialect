// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A trait, an impl and a proof are each closed: their bodies read only what
// their own blocks compute or take as arguments. A proof naming a value the
// module computes, a method naming one, and a method naming a value its
// sibling computes are refused.

!P = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!P]>) {}
trait.trait private @A(%self: !trait.claim<@A[!P]>) {}
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) {}
trait.impl private @A_gen(%self: !trait.claim<@A[!P]>, %b: !trait.claim<@B[!P]>) {}
%outside = trait.witness @B_i64 for @B[i64]
// expected-note @+1 {{required by region isolation constraints}}
trait.proof private @p {
  // expected-error @+1 {{using value defined outside the region}}
  %d = trait.derive @A[i64] from @A_gen given(%outside) : (!trait.claim<@B[i64] by @B_i64>)
  trait.return %d : !trait.claim<@A[i64]>
}

// -----

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) { trait.method @m(!T) -> index }
%g = arith.constant 3 : index
// expected-note @+1 {{required by region isolation constraints}}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>) {
  trait.method @m(%x: !T) -> index {
    // expected-error @+1 {{using value defined outside the region}}
    trait.return %g : index
  }
}

// -----

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
  trait.method @m(!T) -> index
  trait.method @n(!T) -> index
}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>) {
  trait.method @m(%x: !T) -> index {
    %c = arith.constant 3 : index
    trait.return %c : index
  }
  trait.method @n(%x: !T) -> index {
    // expected-error @+1 {{use of undeclared SSA value name}}
    trait.return %c : index
  }
}
