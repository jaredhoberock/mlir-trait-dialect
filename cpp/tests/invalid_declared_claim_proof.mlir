// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A `by @proof` spelled in a function signature is verified where it is
// declared: the named proof must prove the claim it annotates. Here the
// signature claims @T[i64] but names @T_i32_p, which proves @T[i32].

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {}
trait.proof private @T_i32_p {
  %d = trait.derive @T[i32] from @T_i32 given()
  trait.return %d : !trait.claim<@T[i32]>
}

// expected-error @below {{proof @T_i32_p proves '!trait.claim<@T[i32]>', which does not discharge the obligation '!trait.claim<@T[i64]>'}}
func.func private @f(%c: !trait.claim<@T[i64] by @T_i32_p>) {
  return
}
