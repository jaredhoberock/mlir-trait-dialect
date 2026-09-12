// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation with no candidate and an allegation with two satisfiable
// candidates must both be rejected. The diagnostics distinguish the missing
// implementation from incoherent implementations.

!T = !trait.poly<0>

trait.trait private @Absent[!T] {}

trait.trait private @Doubled[!T] {}

trait.impl private @Doubled_wide for @Doubled[i64] {}

trait.impl private @Doubled_narrow for @Doubled[i64] {}

!P = !trait.poly<1>

func.func private @hold(%absent: !trait.claim<@Absent[!P]>,
                %doubled: !trait.claim<@Doubled[!P]>) {
  return
}

func.func @main() {
  %absent = trait.allege @Absent[i64]
  %doubled = trait.allege @Doubled[i64]
  trait.func.call @hold(%absent, %doubled) {type_params = [!trait.poly<1>], type_args = [i64]}
    : (!trait.claim<@Absent[i64]>, !trait.claim<@Doubled[i64]>) -> ()
  return
}

// CHECK-DAG: 'trait.allege' op incoherent impls (multiple satisfiable) for '!trait.claim<@Doubled[i64]>'
// CHECK-DAG: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@Absent[i64]>'
// CHECK: unresolved monomorphic trait.allege after resolve-impls
