// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A blanket impl carries an equality where entry, and the function implementing
// its member passes that entry's parameter to a function. A call of the member
// through a proven, ground receiver reaches an instance of the impl's function
// whose where parameter is replaced by the evidence the proof's derive supplies
// for the entry, cloned from the proof's body, and no reference to the impl's
// arguments remains.
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<1>

trait.trait private @Assoc(!trait.claim<@Assoc[!trait.poly<0>]>) assoc {Out = 0}
trait.impl private @Assoc_i64(%self: !trait.claim<@Assoc[i64]>) bindings {Out = i32} {
}

func.func private @spend(%e: !trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) -> i32 {
  %c = arith.constant 1 : i32
  return %c : i32
}

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!trait.poly<0>]>, !trait.poly<0>) -> i32
trait.impl private @T_impl(%self: !trait.claim<@T[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) members [@T.m -> @T_impl.m] {
}
func.func private @T_impl.m(%self: !trait.claim<@T[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>, %s: !trait.poly<0>) -> i32 {
  %r = trait.func.call @spend(%out) : (!trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) -> i32
  return %r : i32
}
trait.proof private @T_p {
  %p0 = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Out"> resolves i32 by @Assoc_i64
    : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  %d = trait.derive @T[i64] from @T_impl[i64] given(%p0) : (!trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>)
  trait.return %d : !trait.claim<@T[i64]>
}

// The instance contains the proof's evidence for the entry.
// CHECK: func.func private @T_impl.m_{{h[0-9a-f]+}}(%{{.*}}: !trait.claim<@T[i64] by @T_p>, %{{.*}}: i64)
// CHECK: %[[E:.*]] = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Out"> resolves i32 by @Assoc_i64
// CHECK: call @spend_{{.*}}(%[[E]])
// CHECK: return

// CHECK-LABEL: func.func @main
// CHECK: %[[W:.*]] = trait.witness @T_p for @T[i64]
// CHECK: call @T_impl.m_{{h[0-9a-f]+}}(%[[W]], %{{.*}})
func.func @main(%x: i64) -> i32 {
  %w = trait.allege @T[i64]
  %r = trait.func.call @T.m(%w, %x) : (!trait.claim<@T[i64]>, i64) -> i32
  return %r : i32
}
