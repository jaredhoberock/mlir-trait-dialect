// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// An equality claim's endpoints receive only the generic-keyed part of a clone's
// substitution -- the variable bindings -- never a projection resolution or a
// module lookup, because a witness verifier requires the endpoints to stay a
// single-substitution instance of the witness's own equality. So a monomorphic
// clone whose equality endpoint is a ground projection keeps that projection
// spelled, where the same projection standing bare on a parameter would resolve
// to its impl's associated type.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<1>

trait.trait private @Assoc[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Assoc_i64 for @Assoc[i64] { trait.assoc_type @Out = i32 }

// The clone binds S := i64. The equality endpoint is variable-substituted but
// not resolved: it stays proj<@Assoc[i64], "Out">, not i32.
// CHECK: func.func private @tpl_
// CHECK-SAME: !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
func.func private @tpl(%c: !trait.claim<!trait.proj<@Assoc[!S], "Out"> = i32>, %x: !S) -> !S {
  return %x : !S
}

// CHECK-LABEL: func.func @main
// CHECK: call @tpl_
func.func @main(%x: i64, %e: !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>) -> i64 {
  %r = trait.func.call @tpl(%e, %x) {type_params = [!trait.poly<1>], type_args = [i64]} : (!trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>, i64) -> i64
  return %r : i64
}
