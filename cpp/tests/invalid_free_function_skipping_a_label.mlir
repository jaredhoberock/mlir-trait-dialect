// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A free function's labels are its parameters' positions, so a signature
// spelling labels 0 and 2 binds a parameter at label 1 that no call can
// determine. The fault is the callee's, and it is named there.

// expected-error @below {{labels its type parameters 0, 1, ... by position, and its signature skips '!trait.poly<1>'}}
func.func private @g(%x: !trait.poly<0>, %y: !trait.poly<2>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
func.func @main() -> i64 {
  %a = arith.constant 7 : i64
  %b = arith.constant 0 : i32
  %r = trait.func.call @g(%a, %b) : (i64, i32) -> i64
  return %r : i64
}
