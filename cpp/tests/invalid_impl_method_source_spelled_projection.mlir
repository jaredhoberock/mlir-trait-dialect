// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' 2>&1 | FileCheck %s

// The net for impl_method_source_spelled_projection.mlir: an impl member that
// keeps a SOURCE-spelled projection is checked through the impl's own bindings,
// not tolerated. Here the trait requires the second parameter to be Self (i32
// for this impl), but the member spells it as @Container[i32]::Elem, which the
// impl binds to i64. Resolving the source spelling through the impl's bindings
// exposes the disagreement as a located hard failure -- the binding-resolution
// path carries the weight, with no lenient projection-vs-concrete crossing to
// absorb it.

!S = !trait.poly<0>

trait.trait private @Container(!trait.claim<@Container[!S]>) assoc {Elem = 0} members [@Container.id]
func.func private @Container.id(!trait.claim<@Container[!S]>, !S, !S) -> !S

// CHECK: member @Container.id has incompatible signature: expected '(i32, i32) -> i32' but found '(i32, !trait.proj<@Container[i32], "Elem">) -> i32'
trait.impl private @Container_impl(%self_claim: !trait.claim<@Container[i32]>) bindings {Elem = i64} members [@Container.id -> @Container_impl.id] {
}
func.func private @Container_impl.id(%self_claim: !trait.claim<@Container[i32]>, %self: i32, %e: !trait.proj<@Container[i32], "Elem">) -> i32 {
  return %self : i32
}
