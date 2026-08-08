// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The congruence closure's constructor identity carries every part of a type
// that is not a type child. Two types share a constructor only when they differ
// solely in their children, so a coerce that drifts a function type's arity
// split, a vector's shape, or a memref's shape and element -- none of which are
// children -- is refused even with the elementwise equalities cited.

// A function type's inputs/results split is part of its constructor: moving the
// argument to the result is not a child rewrite, so with no evidence at all the
// two are distinct.
func.func @fn_arity(%g: (i64) -> ()) {
  // expected-error @below {{are not equal under the cited equalities}}
  %h = trait.coerce %g : (i64) -> () to () -> (i64)
  return
}

// -----

// A vector's shape is part of its constructor: vector<4xi64> and vector<8xi64>
// share an element type but not a constructor.
func.func @vector_shape(%v: vector<4xi64>) {
  // expected-error @below {{are not equal under the cited equalities}}
  %w = trait.coerce %v : vector<4xi64> to vector<8xi64>
  return
}

// -----

// A memref's shape is part of its constructor, so even citing i64 = i32 for the
// element does not make memref<4xi64> and memref<8xi32> congruent.
func.func @memref_shape(%m: memref<4xi64>, %e: !trait.claim<i64 = i32>) {
  // expected-error @below {{are not equal under the cited equalities}}
  %n = trait.coerce %m : memref<4xi64> to memref<8xi32> via (%e) : (!trait.claim<i64 = i32>)
  return
}
