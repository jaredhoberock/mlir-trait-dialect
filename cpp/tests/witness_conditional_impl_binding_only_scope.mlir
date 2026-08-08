// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --check-prefix=REFUSE

// A conditional impl whose where-bound has no satisfying impl for the
// instantiation, cited by a witness proj_resolve whose binding is nonetheless
// correct and consumed by a trait.coerce. @Has_tuple binds @Has[tuple<!U>]::Out
// to i64 and requires @X[!U]; for !U = i32 the module carries no impl of @X[i32],
// so the impl's requirement is unsatisfied. The proj_resolve certificate binds
// the redex to i64, which is the impl's true binding.
//
// This row pins both halves of the proj-resolve audit's binding-only scope.
// Plain verification accepts: the seam audit judges the citation's binding and
// does not reach the cited impl's requirements. The monomorphize-trait pass
// refuses loudly: the equality claim is monomorphic and no solver ever
// discharged it, so the pipeline backstop reports it after
// instantiate-monomorphs. When the audit is later strengthened to demand the
// cited impl's requirements be discharged by the witness's premises, the first
// RUN line's accept must become a refusal.

!U = !trait.poly<0>

trait.trait @X[!U] {}

trait.trait @Has[!U] {
  trait.assoc_type @Out
}

trait.impl @Has_tuple for @Has[tuple<!U>] where [@X[!U]] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @f
// CHECK: trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple
// CHECK: trait.coerce %arg0 : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via
func.func @f(%v: !trait.proj<@Has[tuple<i32>], "Out">) -> i64 {
  // REFUSE: error: unproven monomorphic claim{{.*}}after instantiate-monomorphs
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
