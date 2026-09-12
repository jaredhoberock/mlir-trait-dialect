// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// The trait reads each method's signature to check that its result parameters
// are determined by its inputs. A method's own invariants are verified after its
// parent's, so the signature is read as an attribute that may be anything: one
// that is not a function type is refused at the method instead of aborting the
// read.

// RUN: mlir-opt %s -verify-diagnostics

trait.trait private @Tr[!trait.poly<0>] {
  // expected-error@+1 {{'func.func' op requires a function type in its 'function_type' attribute}}
  "func.func"() <{sym_name = "m", function_type = i64}> ({}) : () -> ()
}
