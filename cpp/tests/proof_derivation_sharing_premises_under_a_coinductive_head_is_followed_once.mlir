// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | timeout 60 mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// @h derives @H[i32] over @d0, each @dk derives @Dk[i32] citing @d(k+1)
// twice, and @d29 cites @h twice: thirty diamonds under a coinductive head.
// The height below each @dk counts @h zero while @h stands on the chain above
// it, so it is kept with @h and read again on the second citation while @h
// still stands there; each pair is followed once, where reading no such height
// would follow each 2^k times. The derivation is far shallower than the
// obligation limit and verifies. The timeout is a tripwire for the walk
// growing exponentially again, many times what the walk takes.

// CHECK: trait.proof private @h

trait.trait private @H(%self: !trait.claim<@H[!trait.poly<0>]>) {}
// REPEAT 0 29: trait.trait private @D{k}(%self: !trait.claim<@D{k}[!trait.poly<0>]>) {}
trait.impl private @H_i32(%self: !trait.claim<@H[i32]>, %d: !trait.claim<@D0[i32]>) {}
// REPEAT 0 28: trait.impl private @D{k}_i32(%self: !trait.claim<@D{k}[i32]>, %a: !trait.claim<@D{k+1}[i32]>, %b: !trait.claim<@D{k+1}[i32]>) {}
trait.impl private @D29_i32(%self: !trait.claim<@D29[i32]>, %a: !trait.claim<@H[i32]>, %b: !trait.claim<@H[i32]>) {}
trait.proof private @h {
  %d = trait.witness @d0 for @D0[i32]
  %p = trait.derive @H[i32] from @H_i32 given(%d) : (!trait.claim<@D0[i32] by @d0>)
  trait.return %p : !trait.claim<@H[i32]>
}
// REPEAT 0 28: trait.proof private @d{k} { %a = trait.witness @d{k+1} for @D{k+1}[i32] %b = trait.witness @d{k+1} for @D{k+1}[i32] %p = trait.derive @D{k}[i32] from @D{k}_i32 given(%a, %b) : (!trait.claim<@D{k+1}[i32] by @d{k+1}>, !trait.claim<@D{k+1}[i32] by @d{k+1}>) trait.return %p : !trait.claim<@D{k}[i32]> }
trait.proof private @d29 {
  %a = trait.witness @h for @H[i32]
  %b = trait.witness @h for @H[i32]
  %p = trait.derive @D29[i32] from @D29_i32 given(%a, %b) : (!trait.claim<@H[i32] by @h>, !trait.claim<@H[i32] by @h>)
  trait.return %p : !trait.claim<@D29[i32]>
}
