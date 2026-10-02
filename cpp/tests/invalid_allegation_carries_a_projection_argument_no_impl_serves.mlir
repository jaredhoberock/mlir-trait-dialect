// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s 2>&1 | FileCheck %s

// An allegation over a generic associated type whose argument is a projection
// no impl serves. The binding that proves it never reads its argument, so no
// hop resolves the argument, but the allegation's witness still spells it. The
// stage puts that projection to impl selection, which has no impl of @Arg to
// serve it, and refuses the demand it left standing rather than lowering a type
// no impl gives a meaning.

!S = !trait.poly<0>
!A = !trait.poly<1>

trait.trait private @Arg(%self: !trait.claim<@Arg[!S]>) {
  trait.assoc_type @Out
}

trait.trait private @Gat(%self: !trait.claim<@Gat[!S]>) {
  trait.assoc_type @Item<[!A]>
}
trait.impl private @Gat_i64(%self: !trait.claim<@Gat[i64]>) {
  trait.assoc_type @Item<[!trait.poly<2>]> = f32
}

func.func private @need(!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32>)

// CHECK: instantiate-monomorphs left the demand '!trait.proj<@Arg[i64], "Out">' standing and never served it
func.func @main() {
  %e = trait.allege !trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32
  func.call @need(%e) : (!trait.claim<!trait.proj<@Gat[i64], "Item", [!trait.proj<@Arg[i64], "Out">]> = f32>) -> ()
  return
}
