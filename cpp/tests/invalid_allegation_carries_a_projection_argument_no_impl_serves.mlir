// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s 2>&1 | FileCheck %s

// An allegation over a generic associated type whose argument is a projection
// no impl serves. The binding that proves it never reads its argument, but a
// projection whose own argument names no type names none either, so selection
// refuses the projection, naming why @Arg[i64]::Out names no type, and the
// allegation stands unproven rather than lowering a type no impl gives a
// meaning.

!S = !trait.poly<0>
!A = !trait.poly<1>

trait.trait private @Arg(!trait.claim<@Arg[!S]>) assoc {Out = 0}

trait.trait private @Gat(!trait.claim<@Gat[!S]>) assoc {Item = 1}
trait.impl private @Gat_i64(%self: !trait.claim<@Gat[i64]>) bindings {Item = f32} {
}

func.func private @need(!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32>)

// CHECK: error: 'trait.allege' op no impl with satisfiable assumptions for '!trait.proj<@Arg[i64], "Out">'
// CHECK: error: unproven monomorphic claim '!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32>' after instantiate-monomorphs
func.func @main() {
  %e = trait.allege !trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32
  func.call @need(%e) : (!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32>) -> ()
  return
}
