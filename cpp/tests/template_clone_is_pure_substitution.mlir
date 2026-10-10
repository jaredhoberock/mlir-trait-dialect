// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Cloning a method into a free function substitutes the self binding through the
// body, but an equality claim's endpoints receive the variable bindings alone --
// no projection resolution reaches inside them. The same clone shows
// both rules at once: a bare projection parameter resolves to its impl's
// associated type i32, while the projection standing as an equality endpoint on
// the evidence cloned from the proof keeps its spelling proj<@Assoc[i64], "Out">.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<1>

trait.trait private @Assoc(%self: !trait.claim<@Assoc[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Assoc_i64(%self: !trait.claim<@Assoc[i64]>) { trait.assoc_type @Out = i32 }

func.func private @spend(%e: !trait.claim<!trait.proj<@Assoc[!trait.poly<3>], "Out"> = i32>) -> i32 {
  %c = arith.constant 1 : i32
  return %c : i32
}

trait.trait private @T(%self: !trait.claim<@T[!S]>) {
  trait.method @m(!S, !trait.proj<@Assoc[!S], "Out">) -> i32
}
trait.impl private @T_impl(%self: !trait.claim<@T[!trait.poly<2>]>, %out: !trait.claim<!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32>) {
  trait.method @m(%s: !trait.poly<2>, %p: !trait.proj<@Assoc[!trait.poly<2>], "Out">) -> i32 {
    %r = trait.func.call @spend(%out) : (!trait.claim<!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32>) -> i32
    trait.return %r : i32
  }
}
trait.proof private @T_p {
  %p0 = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Out"> resolves i32 by @Assoc_i64
    : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  %d = trait.derive @T[i64] from @T_impl given(%p0) : (!trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>)
  trait.return %d : !trait.claim<@T[i64]>
}

// The bare projection parameter resolves to i32; the equality endpoint does not.
// CHECK: func.func private @T_impl
// CHECK-SAME: : i32) -> i32
// CHECK: trait.witness proj_resolve {{.*}} : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>

// CHECK-LABEL: func.func @main
// CHECK: call @T_impl
func.func @main(%x: i64, %p: !trait.proj<@Assoc[i64], "Out">) -> i32 {
  %w = trait.allege @T[i64]
  %r = trait.method.call %w @T[i64]::@m(%x, %p) : (i64, !trait.proj<@Assoc[i64], "Out">) -> i32
  return %r : i32
}
