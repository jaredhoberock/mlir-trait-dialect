// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(verify-acyclic-traits)' 2>&1 | FileCheck %s

// The comparison of an impl's member function with its trait's member, made
// where the stage begins, reads both signatures through the impl's own
// bindings. @I binds Item<U> beneath a tuple to its own projection at a larger
// argument, once and then twice, and the third @I's function takes a plain
// type where the trait's member spells the projection: what a binding spells beneath its root is read nested in the
// projection's reading, bounded by the depth limit, so each reading stops
// there and the impl is refused rather than recursing until the stack is gone.

// CHECK: :[[# @LINE + 6]]:1: error: 'trait.impl' op projection normalization did not converge
// CHECK: :[[# @LINE + 16]]:1: error: 'trait.impl' op projection normalization did not converge
// CHECK: :[[# @LINE + 26]]:1: error: 'trait.impl' op projection normalization did not converge

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) assoc {Item = 1} members [@A.m]
func.func private @A.m(!trait.claim<@A[!trait.poly<0>]>, !trait.proj<@A[!trait.poly<0>], "Item", [i64]>) -> i64
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>} members [@A.m -> @I.m] {
}
func.func private @I.m(%s: !trait.claim<@A[i32]>, %x: !trait.proj<@A[i32], "Item", [i64]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}

// -----

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) assoc {Item = 1} members [@A.m]
func.func private @A.m(!trait.claim<@A[!trait.poly<0>]>, !trait.proj<@A[!trait.poly<0>], "Item", [i64]>) -> i64
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>} members [@A.m -> @I.m] {
}
func.func private @I.m(%s: !trait.claim<@A[i32]>, %x: !trait.proj<@A[i32], "Item", [i64]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}

// -----

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) assoc {Item = 1} members [@A.m]
func.func private @A.m(!trait.claim<@A[!trait.poly<0>]>, !trait.proj<@A[!trait.poly<0>], "Item", [i64]>) -> ()
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>} members [@A.m -> @I.m] {
}
func.func private @I.m(%s: !trait.claim<@A[i32]>, %x: i64) -> () {
  return
}
