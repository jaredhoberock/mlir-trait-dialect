// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// The trait's self parameter and the impl's own parameter are both label 0 --
// every trait's self is, and an impl spells its parameters from label 0 up --
// and the impl leaves its member to the trait's default. The default is cut at
// the receiver's argument the call states, `tuple<i32>`, and stamped once;
// reading the stamped term again as though it were the trait's own type would
// wrap another tuple around it every time, so the clone here is one tuple deep
// and no deeper.

trait.trait private @Tr(!trait.claim<@Tr[!trait.poly<0>]>) members [default @Tr.method]
func.func private @Tr.method(%self_claim: !trait.claim<@Tr[!trait.poly<0>]>, %self: !trait.poly<0>) -> !trait.poly<0> {
  return %self : !trait.poly<0>
}

trait.impl private @Tr_impl(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>) members [@Tr.method -> @Tr.method] {
}

func.func @main() -> i32 {
  %c = arith.constant 0 : i32
  %v = builtin.unrealized_conversion_cast %c : i32 to tuple<i32>
  %w = trait.allege @Tr[tuple<i32>]
  %r = trait.func.call @Tr.method(%w, %v) : (!trait.claim<@Tr[tuple<i32>]>, tuple<i32>) -> tuple<i32>
  %o = builtin.unrealized_conversion_cast %r : tuple<i32> to i32
  return %o : i32
}

// CHECK-LABEL: func.func private @Tr.method_h
// CHECK-SAME: (%{{.*}}: tuple<i32>) -> tuple<i32>
// CHECK: return %{{.*}} : tuple<i32>
// CHECK-LABEL: func.func @main()
// CHECK: call @Tr.method_h
// CHECK-SAME: (tuple<i32>) -> tuple<i32>
// CHECK-NOT: tuple<tuple
