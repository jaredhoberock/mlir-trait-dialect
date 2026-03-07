// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Test trait.proj.cast case 3: projection -> projection.
//
// Two traits with the same associated type binding. A value typed as
// one projection can be cast to the other.
//
// trait A { type Assoc; }
// trait B { type Assoc; }
// impl A for i64 { type Assoc = i1; }
// impl B for i64 { type Assoc = i1; }

!T = !trait.poly<0>

trait.trait @A[!T] {
  trait.assoc_type @Assoc
}

trait.trait @B[!T] {
  trait.assoc_type @Assoc
}

trait.impl @A_i64 for @A[i64] {
  trait.assoc_type @Assoc = i1
}

trait.impl @B_i64 for @B[i64] {
  trait.assoc_type @Assoc = i1
}

trait.proof @A_proof proves @A_i64 for @A[i64] given []
trait.proof @B_proof proves @B_i64 for @B[i64] given []

// CHECK-LABEL: func.func @proj_to_proj
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: !trait.proj
// CHECK: return %{{.*}} : i1
func.func @proj_to_proj() -> i1 {
  %v = arith.constant true
  %a_claim = trait.witness @A_proof for @A[i64]
  %b_claim = trait.witness @B_proof for @B[i64]

  // Concrete -> A projection
  %a_proj = trait.proj.cast %v, %a_claim
    : i1 to !trait.proj<@A[i64], "Assoc"> claim !trait.claim<@A[i64] by @A_proof>

  // A projection -> B projection (proj -> proj, case 3)
  // The claim matches the result projection's trait application.
  %b_proj = trait.proj.cast %a_proj, %b_claim
    : !trait.proj<@A[i64], "Assoc"> to !trait.proj<@B[i64], "Assoc"> claim !trait.claim<@B[i64] by @B_proof>

  // B projection -> concrete
  %result = trait.proj.cast %b_proj, %b_claim
    : !trait.proj<@B[i64], "Assoc"> to i1 claim !trait.claim<@B[i64] by @B_proof>

  return %result : i1
}
