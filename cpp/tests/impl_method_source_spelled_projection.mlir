// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' | FileCheck %s

// An impl's member function may keep its SOURCE form: a parameter or
// result written in terms of Self::Assoc lowers as a projection naming the
// impl's own trait application. The comparison with the trait's member where
// the stage begins resolves it through the impl's associated type bindings
// rather than requiring the pre-resolved form.
//
// The binding-resolution path carries the check alone: the source-spelled
// projection and the trait's resolved requirement meet at the same grade, with
// no appeal to a lenient projection-vs-concrete rule (this scenario raises no
// residual-tolerance accept). invalid_impl_method_source_spelled_projection.mlir
// is its net: a source spelling whose binding-resolved form disagrees is a
// located hard failure, never a tolerated crossing.

// CHECK-LABEL: trait private @Container
!S = !trait.poly<0>
trait.trait private @Container(!trait.claim<@Container[!S]>) assoc {Elem = 0} members [@Container.id]
func.func private @Container.id(!trait.claim<@Container[!S]>, !S, !trait.proj<@Container[!S], "Elem">) -> !trait.proj<@Container[!S], "Elem">

// CHECK-LABEL: trait.impl private @Container_impl(%self: !trait.claim<@Container[i32]>
// CHECK: func.func private @Container_impl.id(%{{[^:]*}}: !trait.claim<@Container[i32]>, %{{[^:]*}}: i32, %{{[^:]*}}: !trait.proj<@Container[i32], "Elem">) -> !trait.proj<@Container[i32], "Elem">
trait.impl private @Container_impl(%self_claim: !trait.claim<@Container[i32]>) bindings {Elem = i64} members [@Container.id -> @Container_impl.id] {
}
// The member function's signature keeps the projection instead of i64.
func.func private @Container_impl.id(%self_claim: !trait.claim<@Container[i32]>, %self: i32, %e: !trait.proj<@Container[i32], "Elem">) -> !trait.proj<@Container[i32], "Elem"> {
  return %e : !trait.proj<@Container[i32], "Elem">
}
