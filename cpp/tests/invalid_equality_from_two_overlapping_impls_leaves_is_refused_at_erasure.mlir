// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s

// Two impls of @A both apply to i32, binding @A[i32]::Item to i64 and to i32.
// Each proj-resolve leaf verifies on its own: it cites its impl at arguments
// whose premise it is given. The coerce derives i64 = i32 from the two leaves,
// with no hypothesis in @use, and verifies too: the leaves entail it. Only
// selection sees the overlap, refusing @A[i32] as incoherent, so the leaves and
// the coerce stand refused at erasure's entry, whichever pipeline hands it the
// module: after instantiation, which leaves an equality's endpoints to that
// judgment, or alone, as the lowering driver runs it on a module in which
// nothing is pending. Erasure's entry refuses the callee's hypothesis
// i64 = i32 too, where @g assumes it.

// VERIFY: trait.coerce

trait.trait private @M(%self: !trait.claim<@M[!trait.poly<0>]>) {}
trait.trait private @N(%self: !trait.claim<@N[!trait.poly<0>]>) {}
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @M_i32(%self: !trait.claim<@M[i32]>) {}
trait.impl private @N_i32(%self: !trait.claim<@N[i32]>) {}
trait.impl private @A_m(%self: !trait.claim<@A[!trait.poly<0>]>, %m: !trait.claim<@M[!trait.poly<0>]>) { trait.assoc_type @Item = i64 }
trait.impl private @A_n(%self: !trait.claim<@A[!trait.poly<0>]>, %n: !trait.claim<@N[!trait.poly<0>]>) { trait.assoc_type @Item = i32 }

func.func private @g(%e: !trait.claim<i64 = i32>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// CHECK: error: equality '!trait.claim<i64 = i32>' does not settle before erasure: its sides resolve to 'i64' and 'i32'
// CHECK: error: incoherent impls (multiple satisfiable) for '!trait.proj<@A[i32], "Item">'
// CHECK: error: equality '!trait.claim<!trait.proj<@A[i32], "Item"> = i64>' does not settle before erasure: a side resolves to no type a predicate states
// CHECK: error: equality '!trait.claim<!trait.proj<@A[i32], "Item"> = i32>' does not settle before erasure: a side resolves to no type a predicate states
// CHECK: error: equality '!trait.claim<i64 = i32>' does not settle before erasure: its sides resolve to 'i64' and 'i32'
func.func @use() -> i64 {
  %m = trait.witness @M_i32 for @M[i32]
  %n = trait.witness @N_i32 for @N[i32]
  %w1 = trait.witness proj_resolve !trait.proj<@A[i32], "Item"> resolves i64 by @A_m[i32] given(%m) : (!trait.claim<@M[i32] by @M_i32>) : !trait.claim<!trait.proj<@A[i32], "Item"> = i64>
  %w2 = trait.witness proj_resolve !trait.proj<@A[i32], "Item"> resolves i32 by @A_n[i32] given(%n) : (!trait.claim<@N[i32] by @N_i32>) : !trait.claim<!trait.proj<@A[i32], "Item"> = i32>
  %r = trait.witness refl : !trait.claim<i64 = i64>
  %c = trait.coerce %r : !trait.claim<i64 = i64> to !trait.claim<i64 = i32> via (%w1, %w2) : (!trait.claim<!trait.proj<@A[i32], "Item"> = i64>, !trait.claim<!trait.proj<@A[i32], "Item"> = i32>)
  %v = func.call @g(%c) : (!trait.claim<i64 = i32>) -> i64
  return %v : i64
}
