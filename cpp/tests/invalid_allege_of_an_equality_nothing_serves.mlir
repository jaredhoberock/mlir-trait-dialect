// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s -verify-diagnostics

// An allegation of an equality whose projection no impl serves is a claim
// nothing proves, refused where it stands.

!S = !trait.poly<0>
!G = !trait.poly<1>

trait.trait private @Carry(%self: !trait.claim<@Carry[!S, !G]>) { trait.assoc_type @Payload }

func.func private @need(!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i64>)

func.func @main() {
  // expected-error @+2 {{no impl with satisfiable assumptions for '!trait.claim<@Carry[i64, i8]>'}}
  // expected-error @+1 {{unproven monomorphic claim '!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Carry[i64, i8], "Payload"> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i64>) -> ()
  return
}
