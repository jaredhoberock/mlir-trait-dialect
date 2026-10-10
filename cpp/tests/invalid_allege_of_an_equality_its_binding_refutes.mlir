// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s -verify-diagnostics

// An allegation of an equality whose projection the impl selection chooses
// binds to another type is a claim nothing proves, refused where it stands.

!S = !trait.poly<0>
!T = !trait.poly<1>
!G = !trait.poly<2>

trait.trait private @Carry(%self: !trait.claim<@Carry[!trait.poly<0>, !trait.poly<1>]>) { trait.assoc_type @Payload }
trait.impl private @Carry_any(%self: !trait.claim<@Carry[!trait.poly<0>, !trait.poly<1>]>) {
  trait.assoc_type @Payload = !trait.poly<0>
}

func.func private @need(!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i32>)

func.func @main() {
  // expected-error @+2 {{alleges '!trait.proj<@Carry[i64, i8], "Payload">' = 'i32', and impl selection resolves its sides to 'i64' and 'i32'}}
  // expected-error @+1 {{unproven monomorphic claim '!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i32>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Carry[i64, i8], "Payload"> = i32
  func.call @need(%e) : (!trait.claim<!trait.proj<@Carry[i64, i8], "Payload"> = i32>) -> ()
  return
}
