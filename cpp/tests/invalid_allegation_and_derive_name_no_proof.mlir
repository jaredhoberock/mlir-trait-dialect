// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// An allegation asserts its claim and a derive is its claim's evidence: what
// proves either is what the stage builds in its place, so neither claim names a
// proof, not even one that proves its application.

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) {}
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {}
func.func @f() {
  // expected-error @below {{result #0 must be a '!trait.claim' naming no proof, but got '!trait.claim<@T[i32] by @T_i32>'}}
  %a = "trait.allege"() : () -> !trait.claim<@T[i32] by @T_i32>
  return
}

// -----

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) {}
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {}
func.func @f() {
  // expected-error @below {{result #0 must be a '!trait.claim' of a trait application naming no proof, but got '!trait.claim<@T[i32] by @T_i32>'}}
  %d = "trait.derive"() <{impl = @T_i32, impl_args = #trait.impl_args<[]>}> : () -> !trait.claim<@T[i32] by @T_i32>
  return
}

// -----

// In a proof's body, where the derive is the decision the proof records.

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) {}
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {}
trait.impl private @T_i32_too(%self: !trait.claim<@T[i32]>) {}
trait.proof private @p {
  // expected-error @below {{result #0 must be a '!trait.claim' of a trait application naming no proof, but got '!trait.claim<@T[i32] by @T_i32_too>'}}
  %d = "trait.derive"() <{impl = @T_i32, impl_args = #trait.impl_args<[]>}> : () -> !trait.claim<@T[i32] by @T_i32_too>
  trait.return %d : !trait.claim<@T[i32] by @T_i32_too>
}
