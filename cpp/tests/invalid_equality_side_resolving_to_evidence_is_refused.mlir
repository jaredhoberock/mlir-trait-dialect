// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s --check-prefix=ERASE

// An impl may bind an associated type to a type carrying evidence -- the tuple
// dialect's mappers bind their claims so -- but an equality relates types, never
// evidence, so a side that resolves to one is no side of an equality, though
// its endpoints as written name no claim. Selection reads such a side and
// refuses the equality where it is judged: an allegation where instantiation
// would prove it, a hypothesis at erasure's entry.

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }

func.func @alleged() -> i64 {
  // ERASE: error: unproven monomorphic claim '!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">>' before erasure
  // expected-error @below {{'trait.allege' op an equality relates types, and one side of '!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">>' resolves to 'tuple<!trait.claim<@Mark[i32]>>', which carries a claim}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">
  %k = arith.constant 7 : i64
  return %k : i64
}

// -----

trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@Mark[i32]>> }

// ERASE: error: equality '!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">>' does not settle before erasure: its sides resolve to 'tuple<!trait.claim<@Mark[i32]>>' and 'tuple<!trait.claim<@Mark[i32]>>', and an equality relates no claim
// expected-error @below {{equality '!trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">>' does not settle before erasure: its sides resolve to 'tuple<!trait.claim<@Mark[i32]>>' and 'tuple<!trait.claim<@Mark[i32]>>', and an equality relates no claim}}
func.func @assumed(%h: !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.proj<@Wrap[i32], "Item">>) -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}
