// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=IMPL
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=DEFAULT

// A member call reaches the function its receiver's impl pairs with the member
// at the evidence the receiver carries, whatever a coercion spells it as: the
// impl's function at the application the proof's root proves, a default at
// the receiver's proof. A receiver coerced to another written form of @Get[i64]
// and one written at @Get[i64] reach one instance of each.

// IMPL-COUNT-1:    func.func private @Get_gen.get_h{{[0-9a-f]+}}(%{{.*}}: !trait.claim<@Get[i64] by @P>) -> i64
// IMPL-NOT:        func.func private @Get_gen.get_h
// DEFAULT-COUNT-1: func.func private @Get.twice_h{{[0-9a-f]+}}(%{{.*}}: !trait.claim<@Get[i64] by @P>) -> i64
// DEFAULT-NOT:     func.func private @Get.twice_h

!A = !trait.poly<0>
trait.trait private @Fold(!trait.claim<@Fold[!A]>) assoc {Item = 0}
trait.impl private @Fold_i32(%self: !trait.claim<@Fold[i32]>) bindings {Item = i64} {
}
trait.trait private @Get(!trait.claim<@Get[!A]>) members [@Get.get, default @Get.twice]
func.func private @Get.get(!trait.claim<@Get[!A]>) -> i64
func.func private @Get.twice(%self: !trait.claim<@Get[!A]>) -> i64 {
  %a = trait.func.call @Get.get(%self) : (!trait.claim<@Get[!A]>) -> i64
  %b = arith.addi %a, %a : i64
  return %b : i64
}
trait.trait private @Mark(!trait.claim<@Mark[!A]>)
trait.impl private @Mark_i64(%self: !trait.claim<@Mark[i64]>) {
}
trait.impl private @Get_gen(%self: !trait.claim<@Get[!A]>, %mark: !trait.claim<@Mark[!A]>) members [@Get.get -> @Get_gen.get, @Get.twice -> @Get.twice] {
}
func.func private @Get_gen.get(%self: !trait.claim<@Get[!A]>, %mark: !trait.claim<@Mark[!A]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.proof private @P {
  %m = trait.witness @Mark_i64 for @Mark[i64]
  %d = trait.derive @Get[i64] from @Get_gen[i64] given(%m) : (!trait.claim<@Mark[i64] by @Mark_i64>)
  trait.return %d : !trait.claim<@Get[i64]>
}
func.func @main() -> i64 {
  %item = trait.witness proj_resolve !trait.proj<@Fold[i32], "Item"> resolves i64 by @Fold_i32 : !trait.claim<!trait.proj<@Fold[i32], "Item"> = i64>
  %p = trait.witness @P for @Get[i64]
  %q = trait.coerce %p : !trait.claim<@Get[i64] by @P> to !trait.claim<@Get[!trait.proj<@Fold[i32], "Item">]> via (%item) : (!trait.claim<!trait.proj<@Fold[i32], "Item"> = i64>)
  %c = trait.func.call @Get.get(%q) : (!trait.claim<@Get[!trait.proj<@Fold[i32], "Item">]>) -> i64
  %d = trait.func.call @Get.get(%p) : (!trait.claim<@Get[i64] by @P>) -> i64
  %e = trait.func.call @Get.twice(%q) : (!trait.claim<@Get[!trait.proj<@Fold[i32], "Item">]>) -> i64
  %f = trait.func.call @Get.twice(%p) : (!trait.claim<@Get[i64] by @P>) -> i64
  %s = arith.addi %c, %d : i64
  %t = arith.addi %e, %f : i64
  %u = arith.addi %s, %t : i64
  return %u : i64
}
