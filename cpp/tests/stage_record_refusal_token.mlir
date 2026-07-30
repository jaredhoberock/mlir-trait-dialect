// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Impl selection misses a unique satisfiable candidate two ways: @Absent has no
// impl at all, and @Doubled has two. Neither allegation can be proven, so the
// module does not compile; the counts tell the two misses apart, because
// whether a later round could change the answer depends on which it was -- a
// generator can still supply what @Absent lacks, while @Doubled can only ever
// gain more candidates.
//
// The recorded fact does not tell them apart: both refusals render as the same
// token. Which arm a refusal carries decides whether it may be retried, and a
// fact that moved with that could not tell a change of retry policy apart from
// a change of what the stage resolved.

!T = !trait.poly<0>

trait.trait @Absent[!T] {}

trait.trait @Doubled[!T] {}

trait.impl @Doubled_wide for @Doubled[i64] {}

trait.impl @Doubled_narrow for @Doubled[i64] {}

!P = !trait.poly<1>

func.func @hold(%absent: !trait.claim<@Absent[!P]>,
                %doubled: !trait.claim<@Doubled[!P]>) {
  return
}

func.func @main() {
  %absent = trait.allege @Absent[i64]
  %doubled = trait.allege @Doubled[i64]
  trait.func.call @hold(%absent, %doubled)
    : (!trait.claim<@Absent[i64]>, !trait.claim<@Doubled[i64]>) -> ()
  return
}

// CHECK-DAG: trait-stage-record fact impl #trait<application@Absent[i64]> = refused
// CHECK-DAG: trait-stage-record fact impl #trait<application@Doubled[i64]> = refused
// CHECK: trait-stage-record digest value={{.*}} selected-impls=0 refusals-no-candidate=1 refusals-ambiguous=1
