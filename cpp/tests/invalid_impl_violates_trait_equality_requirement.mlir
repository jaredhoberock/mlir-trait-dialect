// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @FoldFn requires Self::Output = Self, but the impl for i32 binds Output to
// i64, so no evidence makes Output and i32 one type and the impl can only
// allege its requirement. @use reads the requirement off a claim the impl
// proves and coerces through it; at i32 the read is replaced by the impl's
// allegation, which selection refuses, and the instance is refused rather than
// run with an i64 read as an i32.

// CHECK: error: 'trait.allege' op alleges '!trait.proj<@FoldFn[i32], "Output">' = 'i32', and impl selection resolves its sides to 'i64' and 'i32'
// CHECK: note: called from

!S = !trait.poly<0>
trait.trait private @FoldFn(%self: !trait.claim<@FoldFn[!S]>) -> !trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S> {
  trait.assoc_type @Output
  trait.method @make() -> !trait.proj<@FoldFn[!S], "Output">
}
trait.impl private @FoldFn_bad(%self: !trait.claim<@FoldFn[i32]>) {
  trait.assoc_type @Output = i64
  %req0 = trait.allege !trait.proj<@FoldFn[i32], "Output"> = i32
  trait.method @make() -> i64 {
    %c = arith.constant 5 : i64
    trait.return %c : i64
  }
  trait.return %req0 : !trait.claim<!trait.proj<@FoldFn[i32], "Output"> = i32>
}
func.func private @use(%f: !trait.claim<@FoldFn[!S]>) -> !S {
  %e = trait.project %f[0] : !trait.claim<@FoldFn[!S]> -> !trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S>
  %v = trait.method.call %f @FoldFn[!S]::@make() : () -> !trait.proj<@FoldFn[!S], "Output">
  %r = trait.coerce %v : !trait.proj<@FoldFn[!S], "Output"> to !S via (%e) : (!trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S>)
  return %r : !S
}
func.func @main() -> i32 {
  %w = trait.witness @FoldFn_bad for @FoldFn[i32]
  %r = trait.func.call @use(%w) : (!trait.claim<@FoldFn[i32] by @FoldFn_bad>) -> i32
  return %r : i32
}
