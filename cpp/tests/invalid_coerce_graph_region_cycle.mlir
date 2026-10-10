// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A coerce stands in a region that enforces SSA dominance, so the chain of
// coercions defining a value ends. In a module body, a graph region, two coerces
// may each be the other's input, and a walk from either to the evidence beneath
// would never end; the coerce is refused where it stands.

trait.trait private @T(%s: !trait.claim<@T[!trait.poly<0>]>) {}
trait.trait private @A(%s: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @AI(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item = i32 }
%e = trait.witness proj_resolve !trait.proj<@A[i32], "Item"> resolves i32 by @AI : !trait.claim<!trait.proj<@A[i32], "Item"> = i32>
// expected-error @below {{'trait.coerce' op must be in a region that enforces SSA dominance}}
%x = trait.coerce %y : !trait.claim<@T[i32]> to !trait.claim<@T[!trait.proj<@A[i32], "Item">]> via (%e) : (!trait.claim<!trait.proj<@A[i32], "Item"> = i32>)
%y = trait.coerce %x : !trait.claim<@T[!trait.proj<@A[i32], "Item">]> to !trait.claim<@T[i32]> via (%e) : (!trait.claim<!trait.proj<@A[i32], "Item"> = i32>)
