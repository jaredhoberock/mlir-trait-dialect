// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A premise whose evidence is an allegation is one the proof cannot decide:
// what the alleged projection denotes is decided by the impl selection chooses
// for its application. @Tensor_i8 returns an allegation for its requirement,
// and @p would read it into @Vector_blanket's equality premise through a
// projection; a proof body holds no projection, so it is refused where it is
// written. Nothing implements @Foo, so @Foo[i8]::Out is settled by nothing and
// is not i64.

trait.trait private @Foo(!trait.claim<@Foo[!trait.poly<0>]>) assoc {Out = 0}
trait.trait private @Tensor(!trait.claim<@Tensor[!trait.poly<0>]>) -> !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i64>
trait.trait private @Vector(!trait.claim<@Vector[!trait.poly<0>]>) members [@Vector.v]
func.func private @Vector.v(!trait.claim<@Vector[!trait.poly<0>]>) -> i64
trait.impl private @Vector_blanket(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i64>) members [@Vector.v -> @Vector_blanket.v] {
}
func.func private @Vector_blanket.v(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i64>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) {
  %e = trait.allege !trait.proj<@Foo[i8], "Out"> = i64
  trait.return %e : !trait.claim<!trait.proj<@Foo[i8], "Out"> = i64>
}
// expected-error @below {{'trait.proof' op unexpected child op 'trait.project'}}
trait.proof private @p {
  %t = trait.witness @Tensor_i8 for @Tensor[i8]
  %e = trait.project %t[0] : !trait.claim<@Tensor[i8] by @Tensor_i8> -> !trait.claim<!trait.proj<@Foo[i8], "Out"> = i64>
  %d = trait.derive @Vector[i8] from @Vector_blanket[i8] given(%t, %e) : (!trait.claim<@Tensor[i8] by @Tensor_i8>, !trait.claim<!trait.proj<@Foo[i8], "Out"> = i64>)
  trait.return %d : !trait.claim<@Vector[i8]>
}
func.func @main() -> i64 {
  %w = trait.witness @p for @Vector[i8]
  %r = trait.func.call @Vector.v(%w) : (!trait.claim<@Vector[i8] by @p>) -> i64
  return %r : i64
}

// -----

// The same rule where two impls of @Foo bind @Foo[i8] rather than none. Both
// bind @Out to i32, so the impl that applies at i8 is @Vector_b; a proof of
// @Vector_a is a proof of the impl whose premise is false, and a witness of it
// would run @Vector_a's member. The evidence the proof states for that premise
// is refused where it is written.

trait.trait private @Foo(!trait.claim<@Foo[!trait.poly<0>]>) assoc {Out = 0}
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) bindings {Out = i32} {
}
trait.impl private @Foo_i8(%self: !trait.claim<@Foo[i8]>) bindings {Out = i32} {
}
trait.trait private @Tensor(!trait.claim<@Tensor[!trait.poly<0>]>)
trait.trait private @Vector(!trait.claim<@Vector[!trait.poly<0>]>) members [@Vector.v]
func.func private @Vector.v(!trait.claim<@Vector[!trait.poly<0>]>) -> i64
trait.impl private @Vector_a(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i64>) members [@Vector.v -> @Vector_a.v] {
}
func.func private @Vector_a.v(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i64>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
trait.impl private @Vector_b(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i32>) members [@Vector.v -> @Vector_b.v] {
}
func.func private @Vector_b.v(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Foo[!trait.poly<0>], "Out"> = i32>) -> i64 {
  %c = arith.constant 2 : i64
  return %c : i64
}
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) {
}
trait.proof private @p {
  %t = trait.witness @Tensor_i8 for @Tensor[i8]
  // expected-error @below {{impl '@Foo_i8' binds the projection to 'i32', not the certified resolution 'i64'}}
  %e = trait.witness proj_resolve !trait.proj<@Foo[i8], "Out"> resolves i64 by @Foo_i8 : !trait.claim<!trait.proj<@Foo[i8], "Out"> = i64>
  %d = trait.derive @Vector[i8] from @Vector_a[i8] given(%t, %e) : (!trait.claim<@Tensor[i8] by @Tensor_i8>, !trait.claim<!trait.proj<@Foo[i8], "Out"> = i64>)
  trait.return %d : !trait.claim<@Vector[i8]>
}
