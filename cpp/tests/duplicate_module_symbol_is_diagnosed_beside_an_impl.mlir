// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// Two module-level symbols named @dup, beside a trait, an impl and the impl's
// member function, whose signature carries a claim. None of them is a symbol
// table and none of their own verifiers reads a symbol: the claims' trait names
// are read in the module's symbol-use walk, which MLIR begins only once the
// module's own names are checked, so the duplicate is diagnosed by name. A
// verifier that indexed the module before that check would abort on the very
// duplicate standing here, in place of the diagnostic naming it.

// CHECK: error: redefinition of symbol named 'dup'
func.func private @dup() { return }
func.func private @dup() { return }

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!trait.poly<0>]>) -> i64
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) members [@T.m -> @T_i32.m] {
}
func.func private @T_i32.m(%self: !trait.claim<@T[i32]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
