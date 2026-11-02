// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait's member takes an equality whose endpoint is a projection of the
// trait's associated type; an impl binding that type writes its member's
// equality in normal form, the projection replaced by its binding. The two
// are compared modulo normalization, as Rust and GHC compare equality
// constraints, so the impl verifies.

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) assoc {A = 0} members [@T.m]
func.func private @T.m(%self: !trait.claim<@T[!trait.poly<0>]>, %e: !trait.claim<!trait.proj<@T[!trait.poly<0>], "A"> = i64>) -> i64
// CHECK: trait.impl private @I(%self: !trait.claim<@T[i1]>)
// CHECK-NEXT: bindings {A = i64}
trait.impl private @I(%self: !trait.claim<@T[i1]>) bindings {A = i64} members [@T.m -> @I.m] {
}
// CHECK: func.func private @I.m(%{{.*}}: !trait.claim<@T[i1]>, %{{.*}}: !trait.claim<i64 = i64>) -> i64
func.func private @I.m(%self: !trait.claim<@T[i1]>, %e: !trait.claim<i64 = i64>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
