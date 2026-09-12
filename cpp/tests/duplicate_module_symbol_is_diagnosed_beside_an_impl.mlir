// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// Two module-level symbols named @dup, beside an impl whose method body carries
// a claim. The impl is a symbol table of its own, so the verifier reaches it
// first and the claim resolves its trait name in the module around it -- a
// module whose own names have not been checked yet. That read is a scan of the
// module, and a scan reports what it found. Indexing the module instead would
// abort on the very duplicate standing here, in place of the diagnostic naming
// it.

// CHECK: error: redefinition of symbol named 'dup'
func.func private @dup() { return }
func.func private @dup() { return }

trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
trait.impl private @T_i32 for @T[i32] {
  func.func @m() -> i64 {
    %s = trait.assume @T[i32]
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
