// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A witness for @Box[@Gen[i64]::A] backed by a proof of @Box[i64] verifies only
// through the residual tolerance: the committed-build match crosses
// @Gen[i64]::A -- which no impl resolves, since @Gen has none -- against the
// rigid i64, and the module-capable comparison accepts it without a binding.
// Verification succeeds and leaves the witness's unresolved projection spelled
// as written.

// CHECK-LABEL: func.func @main()
// CHECK: trait.witness @Box_i64 for @Box[!trait.proj<@Gen[i64], "A">]
// CHECK-NEXT: return

trait.trait private @Gen[!trait.poly<0>] {
  trait.assoc_type @A
}

trait.trait private @Box[!trait.poly<1>] {}

trait.impl private @Box_i64 for @Box[i64] {}

func.func @main() {
  %w = trait.witness @Box_i64 for @Box[!trait.proj<@Gen[i64], "A">]
  return
}
