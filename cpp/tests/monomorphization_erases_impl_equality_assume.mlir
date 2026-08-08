// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A FoldFn-shaped polymorphic impl asserts its own equality Self::Output = Self
// and binds Output to Self. Its method body mints an equality-arm trait.assume
// anchored on that assumption and feeds it to a trait.coerce that rewrites a
// projection-typed value to the accumulator type. The source verifies and the
// equality assume is present.

// VERIFY: trait.assume !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>

!S = !trait.poly<0>

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
  func.func nested @run(!trait.proj<@FoldFn[!S], "Output">) -> !S
}

trait.impl @FoldFn_gen for @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output = !S
  func.func nested @run(%p: !trait.proj<@FoldFn[!S], "Output">) -> !S {
    %e = trait.assume !trait.proj<@FoldFn[!S], "Output"> = !S
    %r = trait.coerce %p : !trait.proj<@FoldFn[!S], "Output"> to !S via (%e)
      : (!trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S>)
    return %r : !S
  }
}

func.func @main(%pv: !trait.proj<@FoldFn[i32], "Output">) -> i32 {
  %c = trait.allege @FoldFn[i32]
  %r = trait.method.call %c @FoldFn[i32]::@run(%pv)
    : (!trait.proj<@FoldFn[i32], "Output">) -> i32
  return %r : i32
}

// Monomorphizing the call instantiates the impl method for i32. The impl binds
// Output = Self, so the projection Self::Output collapses to i32: the assumed
// equality becomes i32 = i32, the coerce it feeds is an identity and folds, and
// the now-dead pure assume is eliminated -- exactly as an application-arm assume
// erases. The concrete instance is a clean identity function with no trait op.

// CHECK: func.func private @FoldFn_gen_{{.*}}_run_{{.*}}(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
// CHECK: func.func @main(%arg0: i32) -> i32
// CHECK: call @FoldFn_gen_{{.*}}_run_
// CHECK-NOT: trait.assume
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.claim
