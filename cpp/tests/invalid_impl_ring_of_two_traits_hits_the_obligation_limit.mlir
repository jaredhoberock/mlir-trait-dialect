// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Two traits whose impls require one another in a ring, the second asking for
// the first at tuple<T>: selection asks about @P1[i32], @P2[i32],
// @P1[tuple<i32>], and around again forever. Every frame is a new
// application, and consecutive frames name different traits, so what stops
// the descent is how many frames stand on the chain and not how often any one
// trait recurs among them: the ring is refused 128 frames deep, where a count
// of one trait's recurrences would stand twice as deep.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@P1[i32]
// CHECK: note: required by {{.*}}@P2[i32]
// CHECK: note: required by {{.*}}@P1[tuple<i32>]
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @P1(!trait.claim<@P1[!trait.poly<0>]>) members [@P1.m]
func.func private @P1.m(!trait.claim<@P1[!trait.poly<0>]>) -> i64
trait.trait private @P2(!trait.claim<@P2[!trait.poly<0>]>) members [@P2.m]
func.func private @P2.m(!trait.claim<@P2[!trait.poly<0>]>) -> i64
trait.impl private @P1_all(%self: !trait.claim<@P1[!trait.poly<0>]>, %p2: !trait.claim<@P2[!trait.poly<0>]>) members [@P1.m -> @P1_all.m] {
}
func.func private @P1_all.m(%self: !trait.claim<@P1[!trait.poly<0>]>, %p2: !trait.claim<@P2[!trait.poly<0>]>) -> i64 {
  %v = trait.func.call @P2.m(%p2) : (!trait.claim<@P2[!trait.poly<0>]>) -> i64
  return %v : i64
}
trait.impl private @P2_all(%self: !trait.claim<@P2[!trait.poly<0>]>, %p1: !trait.claim<@P1[tuple<!trait.poly<0>>]>) members [@P2.m -> @P2_all.m] {
}
func.func private @P2_all.m(%self: !trait.claim<@P2[!trait.poly<0>]>, %p1: !trait.claim<@P1[tuple<!trait.poly<0>>]>) -> i64 {
  %v = trait.func.call @P1.m(%p1) : (!trait.claim<@P1[tuple<!trait.poly<0>>]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.allege @P1[i32]
  %v = trait.func.call @P1.m(%w) : (!trait.claim<@P1[i32]>) -> i64
  return %v : i64
}
