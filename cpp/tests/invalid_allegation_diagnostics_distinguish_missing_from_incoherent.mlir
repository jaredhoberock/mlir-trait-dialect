// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation with no candidate and an allegation with two satisfiable
// candidates must both be rejected. The diagnostics distinguish the missing
// implementation from incoherent implementations.

!T = !trait.poly<0>

trait.trait private @Absent(%self: !trait.claim<@Absent[!T]>) {}

trait.trait private @Doubled(%self: !trait.claim<@Doubled[!T]>) {}

trait.impl private @Doubled_wide(%self: !trait.claim<@Doubled[i64]>) {}

trait.impl private @Doubled_narrow(%self: !trait.claim<@Doubled[i64]>) {}

!P = !trait.poly<1>

func.func private @hold(%absent: !trait.claim<@Absent[!trait.poly<0>]>,
                %doubled: !trait.claim<@Doubled[!trait.poly<0>]>) {
  return
}

func.func @main() {
  %absent = trait.allege @Absent[i64]
  %doubled = trait.allege @Doubled[i64]
  trait.func.call @hold(%absent, %doubled)
    : (!trait.claim<@Absent[i64]>, !trait.claim<@Doubled[i64]>) -> ()
  return
}

// CHECK-DAG: 'trait.allege' op incoherent impls (multiple satisfiable) for '!trait.claim<@Doubled[i64]>'
// CHECK-DAG: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@Absent[i64]>'
// CHECK: unproven monomorphic claim {{.*}} after instantiate-monomorphs
