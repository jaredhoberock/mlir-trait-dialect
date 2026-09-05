// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A standing obligation inside a template is foreign: it is what leaves with the
// template, not work this compilation owes. isPendingExpansion counts it as no
// work, so a module whose only obligation stands in a polymorphic function is not
// pending -- the condition erase may run under.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s

trait.trait private @Tr[!trait.poly<0>] {}

func.func private @tpl(%x: !trait.poly<1>) -> !trait.poly<1> {
  %a = trait.allege @Tr[i64]
  return %x : !trait.poly<1>
}

// CHECK: remark: pending-expansion=false
