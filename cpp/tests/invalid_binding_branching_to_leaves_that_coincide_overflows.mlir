// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | timeout 120 mlir-opt -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// @S binds @C[tuple<T>]'s Out to a pair of @C[T]'s Out at two larger
// arguments, and @Z binds @C[i1]'s to i64 whatever its argument, so the
// leaves of the branching coincide: the type below, twenty deep, reads to a
// tree of 2^20 leaves spelled by twenty-one distinct types, but the reading
// takes a step at each of the 2^21 - 1 distinct projections the branching
// spells on the way. No chain reaches the depth limit and no normal form
// grows; the reading would take more steps than the step budget, and it
// overflows there, named by what it spells as written.

!T = !trait.poly<0>
trait.trait private @C(%s: !trait.claim<@C[!T]>) { trait.assoc_type @Out<[!trait.poly<1>]> }
trait.impl private @Z(%s: !trait.claim<@C[i1]>) { trait.assoc_type @Out<[!trait.poly<0>]> = i64 }
trait.impl private @S(%s: !trait.claim<@C[tuple<!T>]>) { trait.assoc_type @Out<[!trait.poly<1>]> = tuple<!trait.proj<@C[!T], "Out", [tuple<!trait.poly<1>>]>, !trait.proj<@C[!T], "Out", [tuple<!trait.poly<1>, !trait.poly<1>>]>> }
!N0 = i1
// REPEAT 0 19: !N{k+1} = tuple<!N{k}>
// expected-error @below {{overflow evaluating the requirement '!trait.proj<@C[tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<i1>>>>>>>>>>>>>>>>>>>>], "Out", [i64]>': resolving its projections takes more than 262144 steps}}
func.func private @use(!trait.proj<@C[!N20], "Out", [i64]>)

// -----

// The same branching beneath an associated type that ignores its arguments:
// @Drop[i32]'s Out is i64 whatever it is applied to, so every answer reads to
// i64 and the hypothesis below is true, yet reading it takes a step at each
// projection the branching spells beneath @Drop's, and it overflows at the
// step budget.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out<[!U]> }
trait.trait private @Drop(%s: !trait.claim<@Drop[!T]>) { trait.assoc_type @Out<[!trait.poly<1>, !trait.poly<2>]> }
trait.impl private @IDrop(%s: !trait.claim<@Drop[i32]>) { trait.assoc_type @Out<[!trait.poly<0>, !trait.poly<1>]> = i64 }
trait.impl private @I(%s: !trait.claim<@A[tuple<!T>]>) { trait.assoc_type @Out<[!U]> = !trait.proj<@Drop[i32], "Out", [!trait.proj<@A[!T], "Out", [tuple<!U>]>, !trait.proj<@A[!T], "Out", [tuple<!U, !U>]>]> }
trait.impl private @Base(%s: !trait.claim<@A[i1]>) { trait.assoc_type @Out<[!T]> = i64 }
!N0 = i1
// REPEAT 0 19: !N{k+1} = tuple<!N{k}>
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<tuple<i1>>>>>>>>>>>>>>>>>>>>], "Out", [i64]> = i64>': resolving its projections takes more than 262144 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[!N20], "Out", [i64]> = i64>) { return }
