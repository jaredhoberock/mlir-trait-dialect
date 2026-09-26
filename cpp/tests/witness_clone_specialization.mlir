// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// Cloning a template rebuilds each projection-resolution witness it holds
// under the clone's substitution: the endpoints and the arguments move
// together, so the clone of @f at !T = i8 holds the witness at i8 and its
// result is that witness's own equality. The witness survives instantiation
// (the clone returns it), so the rebuilt witness is verified at the instance:
// @S_blanket applies there because the clone's @Tensor[i8] premise, proved by
// @Tensor_i8, binds Shape to i64.

!T = !trait.poly<0>

trait.trait private @Tensor[!T] { trait.assoc_type @Shape }
trait.impl private @Tensor_i8 for @Tensor[i8] { trait.assoc_type @Shape = i64 }
trait.trait private @S[!T] { trait.assoc_type @Out }
trait.impl private @S_blanket for @S[!T]
    where [@Tensor[!T], !trait.proj<@Tensor[!T], "Shape"> = i64] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func private @f(
// CHECK: trait.witness proj_resolve !trait.proj<@S[!trait.poly<0>], "Out"> resolves i64 by @S_blanket[!trait.poly<0> = !trait.poly<0>]
// CHECK-LABEL: func.func private @f_
// CHECK: trait.witness proj_resolve !trait.proj<@S[i8], "Out"> resolves i64 by @S_blanket[!trait.poly<0> = i8] given(%arg0) : (!trait.claim<@Tensor[i8] by @Tensor_i8>) : !trait.claim<!trait.proj<@S[i8], "Out"> = i64>
func.func private @f(%t: !trait.claim<@Tensor[!T]>)
    -> !trait.claim<!trait.proj<@S[!T], "Out"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@S[!T], "Out"> resolves i64 by @S_blanket[!T = !T] given(%t)
    : (!trait.claim<@Tensor[!T]>) : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
  return %e : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
}

func.func @main() {
  %t = trait.allege @Tensor[i8]
  %e = trait.func.call @f(%t) : (!trait.claim<@Tensor[i8]>)
    -> !trait.claim<!trait.proj<@S[i8], "Out"> = i64>
  return
}
