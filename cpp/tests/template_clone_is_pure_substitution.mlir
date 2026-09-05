// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Cloning a method into a free function substitutes the self binding through the
// body, but an equality claim's endpoints receive the variable bindings alone --
// no module lookup resolves the projection inside them. The same clone shows
// both rules at once: a bare projection parameter resolves to its impl's
// associated type i32, while the projection standing as an equality endpoint on
// the synthesized project keeps its spelling proj<@Assoc[i64], "Out">.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<1>

trait.trait private @Assoc[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Assoc_i64 for @Assoc[i64] { trait.assoc_type @Out = i32 }

trait.trait private @T[!S] {
  func.func private @m(!S, !trait.proj<@Assoc[!S], "Out">) -> !trait.claim<!trait.proj<@Assoc[!S], "Out"> = i32>
}
trait.impl private @T_impl for @T[!trait.poly<2>] where [!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32] {
  func.func nested @m(%self: !trait.poly<2>, %p: !trait.proj<@Assoc[!trait.poly<2>], "Out">) -> !trait.claim<!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32> {
    %e = trait.assume !trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32
    return %e : !trait.claim<!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32>
  }
}
trait.proof private @T_p proves @T_impl for @T[i64] given []

// The bare projection parameter resolves to i32; the equality endpoint does not.
// CHECK: func.func private @T_impl
// CHECK-SAME: : i32) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
// CHECK: trait.project %{{.*}} to !trait.proj<@Assoc[i64], "Out"> = i32

// CHECK-LABEL: func.func @main
// CHECK: call @T_impl
func.func @main(%x: i64, %p: !trait.proj<@Assoc[i64], "Out">) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32> {
  %w = trait.allege @T[i64]
  %r = trait.method.call %w @T[i64]::@m(%x, %p) : (i64, !trait.proj<@Assoc[i64], "Out">) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  return %r : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
}
