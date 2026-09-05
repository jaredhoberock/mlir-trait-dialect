// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Instantiation rewrites monomorphic obligations and their calls only where it
// owns the code. A polymorphic function is a template it leaves for the caller
// that clones it, so a proven-shaped obligation and the method call reading it,
// standing inside one, are carried through untouched -- still spelled as an
// allege and a method call -- while the same pair outside a template is proved
// and lowered.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

trait.trait private @Tr[!trait.poly<0>] {
  func.func private @m(!trait.poly<0>) -> i64
}
trait.impl private @Tr_i64 for @Tr[i64] {
  func.func @m(%self: i64) -> i64 { return %self : i64 }
}
trait.proof private @Tr_i64_p proves @Tr_i64 for @Tr[i64] given []

// CHECK: func.func private @tpl
// CHECK: trait.allege @Tr[i64]
// CHECK: trait.method.call {{.*}} @Tr[i64]::@m
func.func private @tpl(%x: !trait.poly<1>, %v: i64) -> i64 {
  %c = trait.allege @Tr[i64]
  %r = trait.method.call %c @Tr[i64]::@m(%v) : (i64) -> i64
  return %r : i64
}

// CHECK-LABEL: func.func @host
// CHECK-NOT: trait.allege
// CHECK-NOT: trait.method.call
// CHECK: call @Tr_i64_m
func.func @host(%v: i64) -> i64 {
  %c = trait.allege @Tr[i64]
  %r = trait.method.call %c @Tr[i64]::@m(%v) : (i64) -> i64
  return %r : i64
}
