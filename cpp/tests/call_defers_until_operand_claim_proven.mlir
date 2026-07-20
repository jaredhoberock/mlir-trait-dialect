// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Pins the deferral guard on call lowering (requireProvenClaimOperands) at both
// call sites it protects. A forwarded proj.cast claim is unproven until the
// cast's projection resolves and its result inherits the input's proof; without
// the guard the greedy driver can specialize a callee while that argument claim
// is still unproven, baking an unprovable claim parameter into the clone whose
// inner method.call then cannot resolve. Proof inheritance on the cast alone
// loses this race; the guard defers the call until the operand claim is proven.
//
// Scenario 1 exercises FuncCallOpLowering: the forwarded claim reaches a
// func.call as an ARGUMENT (not a method-call self). @gen casts its proven
// FoldFn claim into the projection-spelled bound and func.calls @apply_it with
// the (still unproven) cast; the guard defers until specialization for i32
// proves the cast, then the whole chain lowers.
//
// Scenario 2 exercises MethodCallOpLowering's argument gate: the forwarded
// claim reaches a trait.method.call as a NON-SELF argument while the call's
// self claim is already proven. @gen2 casts its proven FoldFn claim and invokes
// @Run::@run through a proven @Run self claim, passing the (still unproven) cast
// as the where-clause argument. The self is proven, so only the argument gate
// can defer this call; without it @Run_i32::@run is specialized with an
// unprovable claim parameter and its inner method.call cannot resolve.

!F = !trait.poly<0>
!G = !trait.poly<0>

trait.trait @FoldFn[!trait.poly<0>, !trait.poly<1>, !trait.poly<2>] {
  func.func private @apply(!trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>
}
trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}
// @Run::@run carries a projection-spelled FoldFn claim as a where-clause
// argument and invokes it, mirroring a bound whose callable is forwarded in.
trait.trait @Run[!trait.poly<0>] {
  func.func private @run(!trait.poly<0>,
                         !trait.claim<@FoldFn[!trait.poly<0>, i64, !trait.proj<@Fold[i1], "Item">]>) -> i64
}
trait.impl @FoldFn_i32 for @FoldFn[i32, i64, i64] {
  func.func @apply(%f: i32, %a: i64, %b: i64) -> i64 {
    %r = arith.addi %a, %b : i64
    return %r : i64
  }
}
trait.impl @Fold_i1 for @Fold[i1] {
  trait.assoc_type @Item = i64
}
trait.impl @Run_i32 for @Run[i32] {
  func.func @run(%self: i32,
                 %ffc: !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[i1], "Item">]>) -> i64 {
    %a = arith.constant 10 : i64
    %b = arith.constant 5 : i64
    %r = trait.method.call %ffc @FoldFn[i32, i64, !trait.proj<@Fold[i1], "Item">]::@apply(%self, %a, %b)
      : (i32, i64, i64) -> i64
    return %r : i64
  }
}

// Scenario 1 callee: generic over G, takes closure g + a PROJECTION-SPELLED
// FoldFn claim (mirrors the real Fold::fold signature at repro.mlir:105), and
// applies it.
func.func @apply_it(%g: !G,
                    %gc: !trait.claim<@FoldFn[!G, i64, !trait.proj<@Fold[i1], "Item">]>) -> i64 {
  %a = arith.constant 10 : i64
  %b = arith.constant 5 : i64
  %r = trait.method.call %gc @FoldFn[!G, i64, !trait.proj<@Fold[i1], "Item">]::@apply(%g, %a, %b)
    : (!G, i64, i64) -> i64
  return %r : i64
}

// Scenario 1 forwarder: has a proven FoldFn claim, forwards it via proj.cast
// into the projection-spelled bound, then FUNC.CALLs @apply_it with the
// (unproven) cast.
func.func @gen(%f: !F, %fc: !trait.claim<@FoldFn[!F, i64, i64]>) -> i64 {
  %fold_c = trait.allege @Fold[i1]
  %cast = trait.proj.cast %fc, %fold_c
    : !trait.claim<@FoldFn[!F, i64, i64]>
    to !trait.claim<@FoldFn[!F, i64, !trait.proj<@Fold[i1], "Item">]>
    by !trait.claim<@Fold[i1]>
  %r = trait.func.call @apply_it(%f, %cast)
    : (!F, !trait.claim<@FoldFn[!F, i64, !trait.proj<@Fold[i1], "Item">]>) -> i64
  return %r : i64
}

func.func @caller() -> i64 {
  %f = arith.constant 7 : i32
  %w = trait.witness @FoldFn_i32 for @FoldFn[i32, i64, i64]
  %r = trait.func.call @gen(%f, %w)
    : (i32, !trait.claim<@FoldFn[i32, i64, i64] by @FoldFn_i32>) -> i64
  return %r : i64
}

// Scenario 2 forwarder: has a proven FoldFn claim and a proven @Run self claim,
// forwards the FoldFn claim via proj.cast, then METHOD.CALLs @Run::@run through
// the proven self, passing the (unproven) cast as the where-clause argument.
func.func @gen2(%f: !F,
                %fc: !trait.claim<@FoldFn[!F, i64, i64]>,
                %rc: !trait.claim<@Run[!F]>) -> i64 {
  %fold_c = trait.allege @Fold[i1]
  %cast = trait.proj.cast %fc, %fold_c
    : !trait.claim<@FoldFn[!F, i64, i64]>
    to !trait.claim<@FoldFn[!F, i64, !trait.proj<@Fold[i1], "Item">]>
    by !trait.claim<@Fold[i1]>
  %r = trait.method.call %rc @Run[!F]::@run(%f, %cast)
    : (!F, !trait.claim<@FoldFn[!F, i64, !trait.proj<@Fold[i1], "Item">]>) -> i64
  return %r : i64
}

func.func @caller2() -> i64 {
  %f = arith.constant 7 : i32
  %fw = trait.witness @FoldFn_i32 for @FoldFn[i32, i64, i64]
  %rw = trait.witness @Run_i32 for @Run[i32]
  %r = trait.func.call @gen2(%f, %fw, %rw)
    : (i32, !trait.claim<@FoldFn[i32, i64, i64] by @FoldFn_i32>,
       !trait.claim<@Run[i32] by @Run_i32>) -> i64
  return %r : i64
}

// Scenario 1: the func.call chain lowers to the concrete impl (arith.addi),
// leaving no trait ops.
// CHECK-NOT: trait.
// CHECK: arith.addi
// CHECK-NOT: trait.
// CHECK-LABEL: func.func @caller
// CHECK: call @gen
// CHECK-NOT: trait.
// Scenario 2: the method.call chain lowers the same way.
// CHECK-LABEL: func.func @caller2
// CHECK: call @gen2
// CHECK-NOT: trait.
