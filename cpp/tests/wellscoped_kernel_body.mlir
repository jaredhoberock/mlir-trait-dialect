// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A kernel is a declaration with a signature, so its body's only source for a
// type argument is that signature, exactly as an ordinary function's is. A
// kernel whose body carries a value through a type parameter its signature does
// not bind is named at the kernel. Nothing else would name it: the two casts
// cancel, so the folder carries the parameter away and the stage exits clean.

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

gpu.module @in_scope_kernels {
  gpu.func @k(%x: !trait.poly<0>) kernel {
    %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<0>
    %b = builtin.unrealized_conversion_cast %a : !trait.poly<0> to i32
    gpu.return
  }
}

// -----

gpu.module @out_of_scope_kernels {
  // expected-error@+1 {{type parameter '!trait.poly<3>' is outside the signature scope of @k}}
  gpu.func @k(%x: i32) kernel {
    // expected-note@+1 {{mentioned here}}
    %a = builtin.unrealized_conversion_cast %x : i32 to !trait.poly<3>
    %b = builtin.unrealized_conversion_cast %a : !trait.poly<3> to i32
    gpu.return
  }
}
