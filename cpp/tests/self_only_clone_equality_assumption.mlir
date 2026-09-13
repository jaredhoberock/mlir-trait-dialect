// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A blanket impl carries an equality where-clause, and one of its methods assumes
// that equality in its body. When a call extracts the method into a free function
// against a proven, ground self, the leading self-proof stands in for the impl:
// each trait.assume of the where-clause equality becomes a trait.project selecting that
// equality from the proven self, and no assume remains in the extracted body.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<1>

trait.trait private @Assoc[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Assoc_i64 for @Assoc[i64] { trait.assoc_type @Out = i32 }

trait.trait private @T[!S] {
  func.func private @m(!S) -> !trait.claim<!trait.proj<@Assoc[!S], "Out"> = i32>
}
trait.impl private @T_impl for @T[!trait.poly<2>] where [!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32] {
  func.func nested @m(%self: !trait.poly<2>) -> !trait.claim<!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32> {
    %e = trait.assume !trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32
    return %e : !trait.claim<!trait.proj<@Assoc[!trait.poly<2>], "Out"> = i32>
  }
}
trait.proof private @T_p proves @T_impl for @T[i64] given []

// The extracted free function projects the equality from the proven self, with
// no assume left behind.
// CHECK: func.func private @T_impl
// CHECK: trait.project %{{.*}}[0] : <@T[i64] by @T_p> -> <!trait.proj<@Assoc[i64], "Out"> = i32>
// CHECK-NOT: trait.assume
// CHECK: return

// CHECK-LABEL: func.func @main
// CHECK: call @T_impl
func.func @main(%x: i64) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32> {
  %w = trait.allege @T[i64]
  %r = trait.method.call %w @T[i64]::@m(%x) : (i64) -> !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  return %r : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
}
