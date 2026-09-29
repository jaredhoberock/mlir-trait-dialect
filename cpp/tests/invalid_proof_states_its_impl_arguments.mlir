// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The proven application is the impl's header at the stated arguments.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A[!S] {}
trait.impl private @A_i32 for @A[i32] {}
trait.trait private @B[!S] {}
trait.impl private @B_tuple for @B[tuple<!U>] where [@A[!U]] {}
// expected-error @below {{impl '@B_tuple' at its stated arguments is an impl of #trait<application@B[tuple<i32>]>, not of #trait<application@B[tuple<i64>]>}}
trait.proof private @p proves @B_tuple[!U = i32] for @B[tuple<i64>] given [@A_i32]

// -----

// One given entry per requirement and where-clause entry.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A[!S] {}
trait.impl private @A_i32 for @A[i32] {}
trait.trait private @C[!S] { trait.assoc_type @Val }
trait.impl private @C_i32 for @C[i32] { trait.assoc_type @Val = i64 }
trait.trait private @B[!S] {}
trait.impl private @B_tuple for @B[tuple<!U>] where [@A[!U], !trait.proj<@C[!U], "Val"> = i64] {}
// expected-error @below {{arity mismatch: impl '@B_tuple' and its trait state 2 requirements and where-clause entries, but found 1 given entries}}
trait.proof private @p proves @B_tuple[!U = i32] for @B[tuple<i32>] given [@A_i32]

// -----

// An application entry is discharged by a symbol.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A[!S] {}
trait.impl private @A_i32 for @A[i32] {}
trait.trait private @B[!S] {}
trait.impl private @B_tuple for @B[tuple<!U>] where [@A[!U]] {}
// expected-error @below {{given entry 0 is unit, and entry 0 is an application a symbol discharges}}
trait.proof private @p proves @B_tuple[!U = i32] for @B[tuple<i32>] given [unit]

// -----

// An equality entry is decided at the proof's claim, so it cites nothing.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @C[!S] { trait.assoc_type @Val }
trait.impl private @C_i32 for @C[i32] { trait.assoc_type @Val = i64 }
trait.trait private @B[!S] {}
trait.impl private @B_tuple for @B[tuple<!U>] where [!trait.proj<@C[!U], "Val"> = i64] {}
// expected-error @below {{given entry 0 is @C_i32, and entry 0 is decided without a symbol, so its given entry is unit}}
trait.proof private @p proves @B_tuple[!U = i32] for @B[tuple<i32>] given [@C_i32]

// -----

// A proof states the arguments its impl's parameters take; a spelling stating
// none is refused where it is read.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A[!S] {}
trait.trait private @B[!S] {}
trait.impl private @B_tuple for @B[tuple<!U>] where [@A[!U]] {}
// expected-error @below {{expected the arguments the impl's parameters take, `[!P = T, ...]`}}
trait.proof private @p proves @B_tuple for @B[tuple<i32>] given [unit]
