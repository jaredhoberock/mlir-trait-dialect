// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An obligation premise spelled through a projection discharges the cited impl's
// assumption modulo an equality premise. @Has_tuple's assumption specializes to
// @X[i32]; the witness supplies @X[!Other[i32]::A] together with the equality
// premise !Other[i32]::A = i32. Verification rewrites the respelled premise by that
// equality to @X[i32] and the assumption is discharged -- the equality premise
// serves as the rewrite modulus, and the two admitted moduli (proof strip and
// equality premises) are the only ones.

!U = !trait.poly<0>

trait.trait @X[!U] {}

trait.trait @Other[!U] {
  trait.assoc_type @A
}

trait.trait @Has[!U] {
  trait.assoc_type @Out
}

trait.impl @X_i32 for @X[i32] {}

trait.impl @Other_i32 for @Other[i32] {
  trait.assoc_type @A = i32
}

trait.impl @Has_tuple for @Has[tuple<!U>] where [@X[!U]] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @f
// CHECK: trait.witness proj_resolve
func.func @f(
    %v: !trait.proj<@Has[tuple<i32>], "Out">,
    %x: !trait.claim<@X[!trait.proj<@Other[i32], "A">]>,
    %e: !trait.claim<!trait.proj<@Other[i32], "A"> = i32>
) -> i64 {
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple given(%x, %e)
    : (!trait.claim<@X[!trait.proj<@Other[i32], "A">]>, !trait.claim<!trait.proj<@Other[i32], "A"> = i32>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
