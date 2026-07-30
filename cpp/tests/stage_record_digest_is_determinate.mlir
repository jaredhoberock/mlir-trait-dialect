// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: (env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -o /dev/null; env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -o /dev/null) 2>&1 | FileCheck %s

// The recorded facts live in maps keyed by pointer, so the order they come out
// in is the order the allocator happened to hand out addresses -- which differs
// between two runs of the same compilation. The module is compiled twice below
// and the two runs must agree on the digest, which is what makes the digest
// usable to compare one version of the stage against another.
//
// The module resolves several applications so that there is an order to get
// wrong: three traits, four impls, and a proof for each application resolved.

!T = !trait.poly<0>

trait.trait @Zero[!T] {}
trait.trait @One[!T] {}
trait.trait @Two[!T] where [
  @Zero[!T],
  @One[!T]
] {}

trait.impl @Zero_i32 for @Zero[i32] {}
trait.impl @Zero_i64 for @Zero[i64] {}
trait.impl @One_i32 for @One[i32] {}
trait.impl @Two_i32 for @Two[i32] {}

!P = !trait.poly<1>

func.func @hold_zero(%zero: !trait.claim<@Zero[!P]>) {
  return
}

func.func @hold_two(%two: !trait.claim<@Two[!P]>) {
  return
}

func.func @main() {
  %zero64 = trait.allege @Zero[i64]
  %zero32 = trait.allege @Zero[i32]
  %two = trait.allege @Two[i32]
  trait.func.call @hold_zero(%zero32) : (!trait.claim<@Zero[i32]>) -> ()
  trait.func.call @hold_zero(%zero64) : (!trait.claim<@Zero[i64]>) -> ()
  trait.func.call @hold_two(%two) : (!trait.claim<@Two[i32]>) -> ()
  return
}

// CHECK: trait-stage-record digest value=[[DIGEST:0x[0-9a-f]+]] selected-impls=4 refusals-no-candidate=0 refusals-ambiguous=0 assumption-facts=4 proofs=4
// CHECK: trait-stage-record digest value=[[DIGEST]] selected-impls=4 refusals-no-candidate=0 refusals-ambiguous=0 assumption-facts=4 proofs=4
