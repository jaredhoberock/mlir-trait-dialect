// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A quantified requirement is an evidence-computing member of its trait. Its type
// arguments are read off the premises a call supplies, and the call's result
// spells the conclusion at them: a result spelled at another argument is
// refused.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(!trait.claim<@Marker[!S]>)
trait.trait private @Has(!trait.claim<@Has[!S]>) assoc {A = 1} members [@Has.requirement_0]
func.func private @Has.requirement_0(!trait.claim<@Has[!S]>, !trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>, %p: !trait.claim<@Marker[i64]>) {
  // expected-error @below {{type mismatch: expected '(!trait.claim<@Has[!trait.poly<2>]>, !trait.claim<@Marker[i64]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i64]>]>' but found '(!trait.claim<@Has[!trait.poly<2>]>, !trait.claim<@Marker[i64]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>'}}
  %m = trait.func.call @Has.requirement_0(%h, %p) : (!trait.claim<@Has[!trait.poly<2>]>, !trait.claim<@Marker[i64]>) -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>
  return
}
