// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A blanket impl carries an equality where entry, and one of its methods passes
// that entry's block argument to a function. When a call extracts the method
// into a free function against a proven, ground self, the entry's argument is
// replaced by the evidence the proof's derive supplies for it, cloned from the
// proof's body, and no reference to the impl's arguments remains.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<1>

trait.trait private @Assoc(%self: !trait.claim<@Assoc[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Assoc_i64(%self: !trait.claim<@Assoc[i64]>) { trait.assoc_type @Out = i32 }

func.func private @spend(%e: !trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) -> i32 {
  %c = arith.constant 1 : i32
  return %c : i32
}

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {
  trait.method @m(!trait.poly<0>) -> i32
}
trait.impl private @T_impl(%self: !trait.claim<@T[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) {
  trait.method @m(%s: !trait.poly<0>) -> i32 {
    %r = trait.func.call @spend(%out) : (!trait.claim<!trait.proj<@Assoc[!trait.poly<0>], "Out"> = i32>) -> i32
    trait.return %r : i32
  }
}
trait.proof private @T_p {
  %p0 = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Out"> resolves i32 by @Assoc_i64
    : !trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>
  %d = trait.derive @T[i64] from @T_impl[i64] given(%p0) : (!trait.claim<!trait.proj<@Assoc[i64], "Out"> = i32>)
  trait.return %d : !trait.claim<@T[i64]>
}

// The extracted free function holds the proof's evidence for the entry.
// CHECK: func.func private @T_impl
// CHECK: %[[E:.*]] = trait.witness proj_resolve !trait.proj<@Assoc[i64], "Out"> resolves i32 by @Assoc_i64
// CHECK: call @spend_{{.*}}(%[[E]])
// CHECK: return

// CHECK-LABEL: func.func @main
// CHECK: call @T_impl
func.func @main(%x: i64) -> i32 {
  %w = trait.allege @T[i64]
  %r = trait.method.call %w @T[i64]::@m(%x) : (i64) -> i32
  return %r : i32
}
