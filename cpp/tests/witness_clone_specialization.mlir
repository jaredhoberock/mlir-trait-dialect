// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// Cloning a template rebuilds each projection-resolution witness it holds
// under the clone's substitution: the endpoints and the premises move
// together, so the clone of @f at !T = i8 holds the witness at i8 and its
// result is that witness's own equality. The witness survives instantiation
// (the clone returns it), so the rebuilt witness is verified at the instance:
// @S_blanket applies there because the clone's premises -- @Tensor[i8], proved
// by @Tensor_i8, and its Shape = i64 -- are the ones its where clause takes.

!T = !trait.poly<0>

trait.trait private @Tensor(!trait.claim<@Tensor[!T]>) assoc {Shape = 0}
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) bindings {Shape = i64} {
}
trait.trait private @S(!trait.claim<@S[!T]>) assoc {Out = 0}
trait.impl private @S_blanket(%self: !trait.claim<@S[!T]>, %tensor: !trait.claim<@Tensor[!T]>, %shape: !trait.claim<!trait.proj<@Tensor[!T], "Shape"> = i64>) bindings {Out = i64} {
}

// CHECK-LABEL: func.func private @f(
// CHECK: trait.witness proj_resolve !trait.proj<@S[!trait.poly<0>], "Out"> resolves i64 by @S_blanket[!trait.poly<0>] given(%arg0, %arg1)
// CHECK-LABEL: func.func private @f_
// CHECK: trait.witness proj_resolve !trait.proj<@S[i8], "Out"> resolves i64 by @S_blanket[i8] given(%arg0, %arg1) : (!trait.claim<@Tensor[i8] by @Tensor_i8>, !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>) : !trait.claim<!trait.proj<@S[i8], "Out"> = i64>
func.func private @f(%t: !trait.claim<@Tensor[!T]>, %sh: !trait.claim<!trait.proj<@Tensor[!T], "Shape"> = i64>)
    -> !trait.claim<!trait.proj<@S[!T], "Out"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@S[!T], "Out"> resolves i64 by @S_blanket[!trait.poly<0>] given(%t, %sh)
    : (!trait.claim<@Tensor[!T]>, !trait.claim<!trait.proj<@Tensor[!T], "Shape"> = i64>)
    : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
  return %e : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
}

func.func @main() {
  %t = trait.allege @Tensor[i8]
  %sh = trait.allege !trait.proj<@Tensor[i8], "Shape"> = i64
  %e = trait.func.call @f(%t, %sh) : (!trait.claim<@Tensor[i8]>, !trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>)
    -> !trait.claim<!trait.proj<@S[i8], "Out"> = i64>
  return
}
