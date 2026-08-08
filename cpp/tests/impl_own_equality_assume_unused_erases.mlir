// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An impl-own equality assumption whose trait.assume is NEVER USED in the method
// body (a marker-body shape). Birth accepts it -- the symbolic endpoint cannot be
// decided against the impl's bindings and defers -- and the assume is a pure op
// that justifies nothing, so it erases completely through monomorphization: the
// concrete instance is a bare identity function with no trait op. An unused
// equality assumption is inert.

// VERIFY: trait.assume !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>

!S = !trait.poly<0>

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
  func.func nested @run(!S) -> !S
}

trait.impl @FoldFn_gen for @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output = !S
  func.func nested @run(%x: !S) -> !S {
    %e = trait.assume !trait.proj<@FoldFn[!S], "Output"> = !S
    return %x : !S
  }
}

func.func @main(%v: i32) -> i32 {
  %c = trait.allege @FoldFn[i32]
  %r = trait.method.call %c @FoldFn[i32]::@run(%v) : (i32) -> i32
  return %r : i32
}

// CHECK: func.func private @FoldFn_gen_{{.*}}_run_{{.*}}(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
// CHECK: func.func @main(%arg0: i32) -> i32
// CHECK-NOT: trait.assume
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.claim
