// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A FoldFn-shaped polymorphic impl takes the equality Out[Self]::Output = Self
// as a where entry, and its method body reads that argument in a trait.coerce
// that rewrites a projection-typed value to the accumulator type.

// VERIFY: trait.impl private @FoldFn_gen(%self: !trait.claim<@FoldFn[!trait.poly<0>]>, %out: !trait.claim<@Out[!trait.poly<0>]>, %output: !trait.claim<!trait.proj<@Out[!trait.poly<0>], "Output"> = !trait.poly<0>>)
// VERIFY: trait.coerce %{{.*}} : !trait.proj<@Out[!trait.poly<0>], "Output"> to !trait.poly<0> via (%output)

!S = !trait.poly<0>

trait.trait private @Out(%self: !trait.claim<@Out[!S]>) {
  trait.assoc_type @Output
}
trait.impl private @Out_i32(%self: !trait.claim<@Out[i32]>) {
  trait.assoc_type @Output = i32
}

trait.trait private @FoldFn(%self: !trait.claim<@FoldFn[!S]>) {
  trait.method @run(!trait.proj<@Out[!S], "Output">) -> !S
}

trait.impl private @FoldFn_gen(%self: !trait.claim<@FoldFn[!S]>, %out: !trait.claim<@Out[!S]>, %output: !trait.claim<!trait.proj<@Out[!S], "Output"> = !S>) {
  trait.method @run(%p: !trait.proj<@Out[!S], "Output">) -> !S {
    %r = trait.coerce %p : !trait.proj<@Out[!S], "Output"> to !S via (%output)
      : (!trait.claim<!trait.proj<@Out[!S], "Output"> = !S>)
    trait.return %r : !S
  }
}

func.func @main(%pv: !trait.proj<@Out[i32], "Output">) -> i32 {
  %c = trait.allege @FoldFn[i32]
  %r = trait.method.call %c @FoldFn[i32]::@run(%pv)
    : (!trait.proj<@Out[i32], "Output">) -> i32
  return %r : i32
}

// Monomorphizing the call instantiates the impl method for i32. Out[i32] binds
// Output to i32, so the projection collapses to i32: the equality argument
// becomes i32 = i32, the coerce it feeds is an identity, and the evidence the
// instance cloned for the argument dies with it. The concrete instance is a
// clean identity function with no trait op.

// CHECK: func.func private @FoldFn_gen_{{h[0-9a-f]+}}_run(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
// CHECK: func.func @main(%arg0: i32) -> i32
// CHECK: call @FoldFn_gen_{{h[0-9a-f]+}}_run
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.claim
