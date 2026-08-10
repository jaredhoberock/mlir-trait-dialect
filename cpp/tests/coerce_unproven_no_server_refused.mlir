// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" 2>&1 | FileCheck %s

// A marked coerce is a promise that some impl will supply its projection's
// ground resolution. When no impl and no generator can serve the projection,
// the promise is unkept: monomorphization leaves the projection standing and
// fails at the existing loud checkpoint. The marker never tolerates a survivor.

trait.trait @Base[!trait.poly<0>] {
  trait.assoc_type @Assoc
}

// CHECK: error: unresolved projection '!trait.proj<@Base[i64], "Assoc">' after instantiate-monomorphs
func.func @use(%x: !trait.proj<@Base[i64], "Assoc">) -> i1 {
  %y = trait.coerce %x : !trait.proj<@Base[i64], "Assoc"> to i1 unproven
  return %y : i1
}
