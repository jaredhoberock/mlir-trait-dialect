// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A parent's verifier runs before its children's, so a child's shape is whatever
// was written when the parent reads it. The trait reads its associated types'
// parameter lists to learn which labels its where clause may mention; an entry
// that is not a type is refused where it stands instead of aborting the read.

// RUN: mlir-opt %s -verify-diagnostics

trait.trait private @Tr[!trait.poly<0>] {
  // expected-error@+1 {{'trait.assoc_type' op type parameter list holds 42 : i64, which is not a type}}
  "trait.assoc_type"() <{sym_name = "A", type_params = [42 : i64]}> : () -> ()
}
