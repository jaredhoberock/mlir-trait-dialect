// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @Tr_i64 takes no parameters and no where entries, so it is its own proof,
// and the evidence it returns for its trait's requirement is a derive of
// itself coerced through its own binding. A derive of such an impl is
// witnessed by the impl's own symbol, as selection names it, and the
// projected requirement, spelled through the binding, is that evidence
// respelled, which the instance key carries back to the impl's header: the
// call through the projected requirement and the call through the witness
// reach one instance of @v rather than two under two names for one evidence.
//
// The checks read the instances the module holds: the defect this pins cuts a
// second copy of @v under a second name for one proof, which runs the same
// code and prints the same 6, so nothing a run observes tells the two apart.

// CHECK-COUNT-1: func.func private @Tr_i64.v_{{h[0-9a-f]+}}()
// CHECK-NOT: func.func private @Tr_i64.v_{{h[0-9a-f]+}}()
// CHECK: func.func @main
// CHECK: call @[[V:Tr_i64.v_h[0-9a-f]+]]()
// CHECK: call @[[V]]()

!S = !trait.poly<0>
trait.trait private @Tr(!trait.claim<@Tr[!S]>) -> !trait.claim<@Tr[!trait.proj<@Tr[!S], "Out">]> assoc {Out = 0} members [@Tr.v]
func.func private @Tr.v(!trait.claim<@Tr[!S]>) -> i64
trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) bindings {Out = i64} members [@Tr.v -> @Tr_i64.v] {
  %own = trait.derive @Tr[i64] from @Tr_i64 given()
  %eq = trait.witness proj_resolve !trait.proj<@Tr[i64], "Out"> resolves i64 by @Tr_i64 : !trait.claim<!trait.proj<@Tr[i64], "Out"> = i64>
  %r = trait.coerce %own : !trait.claim<@Tr[i64]> to !trait.claim<@Tr[!trait.proj<@Tr[i64], "Out">]> via (%eq) : (!trait.claim<!trait.proj<@Tr[i64], "Out"> = i64>)
  trait.return %r : !trait.claim<@Tr[!trait.proj<@Tr[i64], "Out">]>
}
func.func private @Tr_i64.v(%self: !trait.claim<@Tr[i64]>) -> i64 {
  %c = arith.constant 3 : i64
  return %c : i64
}
func.func @main() -> i64 {
  %t = trait.witness @Tr_i64 for @Tr[i64]
  %a = trait.func.call @Tr.v(%t) : (!trait.claim<@Tr[i64] by @Tr_i64>) -> i64
  %o = trait.project %t[0] : !trait.claim<@Tr[i64] by @Tr_i64> -> !trait.claim<@Tr[!trait.proj<@Tr[i64], "Out">]>
  %b = trait.func.call @Tr.v(%o) : (!trait.claim<@Tr[!trait.proj<@Tr[i64], "Out">]>) -> i64
  %s = arith.addi %a, %b : i64
  return %s : i64
}
