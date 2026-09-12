// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// The impl reads each binding's parameter list to judge it against its header's
// parameters. A binding's own invariants are verified after its parent's, so an
// entry that is not a type is refused where it stands instead of aborting the
// read.

// RUN: mlir-opt %s -verify-diagnostics

trait.trait private @Tr[!trait.poly<0>] {
  trait.assoc_type @A<[!trait.poly<1>]>
}

trait.impl private @Tr_i32 for @Tr[i32] {
  // expected-error@+1 {{'trait.assoc_type' op type parameter list holds 42 : i64, which is not a type}}
  "trait.assoc_type"() <{sym_name = "A", bound_type = i32, type_params = [42 : i64]}> : () -> ()
}
