// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A nested module declares @Has and the root does not. The op spelling the
// projection stands there, and the demand it raises is put to selection at the
// root, keyed by a trait name the root cannot resolve. A demand naming a trait
// this scope does not declare has no candidate here, so it is refused as a
// demand no impl satisfies and the spelling it could not serve is reported
// where it stands.

func.func @main() -> i64 {
  %c = arith.constant 0 : i64
  return %c : i64
}
module @inner {
  trait.trait private @Has[!trait.poly<0>] { trait.assoc_type @Out }
  trait.impl private @Has_i32 for @Has[i32] { trait.assoc_type @Out = i64 }
  // expected-error @below {{unresolved projection '!trait.proj<@Has[i32], "Out">' after instantiate-monomorphs}}
  func.func @g(%x: !trait.proj<@Has[i32], "Out">) -> i64 {
    %c = trait.coerce %x : !trait.proj<@Has[i32], "Out"> to i64 unproven
    return %c : i64
  }
}
