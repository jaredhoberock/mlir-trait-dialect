// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// An impl, a proof and a member function are each closed: their bodies read
// only what their own blocks compute or take as arguments. A proof naming a
// value the module computes, a member function naming one, and a member
// function naming a value its sibling computes are refused.

!P = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!P]>)
trait.trait private @A(!trait.claim<@A[!P]>)
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) {
}
trait.impl private @A_gen(%self: !trait.claim<@A[!P]>, %b: !trait.claim<@B[!P]>) {
}
%outside = trait.witness @B_i64 for @B[i64]
// expected-note @+1 {{required by region isolation constraints}}
trait.proof private @p {
  // expected-error @+1 {{using value defined outside the region}}
  %d = trait.derive @A[i64] from @A_gen[i64] given(%outside) : (!trait.claim<@B[i64] by @B_i64>)
  trait.return %d : !trait.claim<@A[i64]>
}

// -----

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.m]
func.func private @A.m(!trait.claim<@A[!T]>, !T) -> index
%g = arith.constant 3 : index
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>) members [@A.m -> @A_gen.m] {
}
// expected-note @+1 {{required by region isolation constraints}}
func.func private @A_gen.m(%self: !trait.claim<@A[!T]>, %x: !T) -> index {
  // expected-error @+1 {{using value defined outside the region}}
  return %g : index
}

// -----

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.m, @A.n]
func.func private @A.m(!trait.claim<@A[!T]>, !T) -> index
func.func private @A.n(!trait.claim<@A[!T]>, !T) -> index
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>) members [@A.m -> @A_gen.m, @A.n -> @A_gen.n] {
}
func.func private @A_gen.m(%self: !trait.claim<@A[!T]>, %x: !T) -> index {
  %c = arith.constant 3 : index
  return %c : index
}
func.func private @A_gen.n(%self: !trait.claim<@A[!T]>, %x: !T) -> index {
  // expected-error @+1 {{use of undeclared SSA value name}}
  return %c : index
}
