// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A quantified requirement is an evidence method of its trait. Its type
// arguments are read off the premises a call supplies, and the call's result
// spells the conclusion at them: a result spelled at another argument is
// refused.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>, %p: !trait.claim<@Marker[i64]>) {
  // expected-error @below {{type mismatch: expected '(!trait.claim<@Marker[i64]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i64]>]>' but found '(!trait.claim<@Marker[i64]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>'}}
  %m = trait.method.call %h @Has[!trait.poly<2>]::@requirement_0(%p) : (!trait.claim<@Marker[i64]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>
  return
}
