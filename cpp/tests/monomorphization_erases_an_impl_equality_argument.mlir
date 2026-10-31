// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A FoldFn-shaped polymorphic impl takes the equality Out[Self]::Output = Self
// as a where entry, and the function implementing its member, which leads with
// the impl's block arguments, reads that argument in a trait.coerce that
// rewrites a projection-typed value to the accumulator type.

// VERIFY: trait.impl private @FoldFn_gen(%self: !trait.claim<@FoldFn[!trait.poly<0>]>, %out: !trait.claim<@Out[!trait.poly<0>]>, %output: !trait.claim<!trait.proj<@Out[!trait.poly<0>], "Output"> = !trait.poly<0>>)
// VERIFY: func.func private @FoldFn_gen.run(%{{.*}}: !trait.claim<@FoldFn[!trait.poly<0>]>, %{{.*}}: !trait.claim<@Out[!trait.poly<0>]>, %[[OUTPUT:arg[0-9]+]]: !trait.claim<!trait.proj<@Out[!trait.poly<0>], "Output"> = !trait.poly<0>>,
// VERIFY: trait.coerce %{{.*}} : !trait.proj<@Out[!trait.poly<0>], "Output"> to !trait.poly<0> via (%[[OUTPUT]])

!S = !trait.poly<0>

trait.trait private @Out(!trait.claim<@Out[!S]>) assoc {Output = 0}
trait.impl private @Out_i32(%self: !trait.claim<@Out[i32]>) bindings {Output = i32} {
}

trait.trait private @FoldFn(!trait.claim<@FoldFn[!S]>) members [@FoldFn.run]
func.func private @FoldFn.run(!trait.claim<@FoldFn[!S]>, !trait.proj<@Out[!S], "Output">) -> !S

trait.impl private @FoldFn_gen(%self: !trait.claim<@FoldFn[!S]>, %out: !trait.claim<@Out[!S]>, %output: !trait.claim<!trait.proj<@Out[!S], "Output"> = !S>) members [@FoldFn.run -> @FoldFn_gen.run] {
}
func.func private @FoldFn_gen.run(%self: !trait.claim<@FoldFn[!S]>, %out: !trait.claim<@Out[!S]>, %output: !trait.claim<!trait.proj<@Out[!S], "Output"> = !S>, %p: !trait.proj<@Out[!S], "Output">) -> !S {
  %r = trait.coerce %p : !trait.proj<@Out[!S], "Output"> to !S via (%output)
    : (!trait.claim<!trait.proj<@Out[!S], "Output"> = !S>)
  return %r : !S
}

func.func @main(%pv: !trait.proj<@Out[i32], "Output">) -> i32 {
  %c = trait.allege @FoldFn[i32]
  %r = trait.func.call @FoldFn.run(%c, %pv) : (!trait.claim<@FoldFn[i32]>, !trait.proj<@Out[i32], "Output">) -> i32
  return %r : i32
}

// Monomorphizing the call instantiates the impl's function for i32. Out[i32] binds
// Output to i32, so the projection collapses to i32: the equality argument
// becomes i32 = i32, the coerce it feeds is an identity, and the evidence the
// instance cloned for the argument dies with it. The concrete instance is a
// clean identity function with no trait op.

// CHECK: func.func private @FoldFn_gen.run_{{h[0-9a-f]+}}(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
// CHECK: func.func @main(%arg0: i32) -> i32
// CHECK: call @FoldFn_gen.run_{{h[0-9a-f]+}}
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.claim
