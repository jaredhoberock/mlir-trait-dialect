// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation states an obligation outstanding until the stage decides it,
// so one nothing reads is no dead op: it stands to be decided, and a false one
// is refused rather than erased unread.

// CHECK: error: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@T[i32]>'

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}

func.func @main() {
  %c = trait.allege @T[i32]
  return
}
