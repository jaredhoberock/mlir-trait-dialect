// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s --check-prefix=ERASE

// @Run_cond applies where @Has[T]::Out = @Other[T]::Out, and at i32 both sides
// read to @Dead[i32]::Out, which no impl serves: the premise is about no type,
// so selection reads it as it reads every equality and @Run_cond is no
// candidate at i32, though the two sides read to one spelling. One verdict
// stands for @Run[i32]::Item wherever it is asked: an equality assumed over it
// does not settle at erasure's entry, and one alleged over it is unproven.

!T = !trait.poly<0>
trait.trait private @Dead(%self: !trait.claim<@Dead[!T]>) { trait.assoc_type @Out }
trait.trait private @Has(%self: !trait.claim<@Has[!T]>) { trait.assoc_type @Out }
trait.impl private @Has_any(%self: !trait.claim<@Has[!T]>) { trait.assoc_type @Out = !trait.proj<@Dead[!T], "Out"> }
trait.trait private @Other(%self: !trait.claim<@Other[!T]>) { trait.assoc_type @Out }
trait.impl private @Other_any(%self: !trait.claim<@Other[!T]>) { trait.assoc_type @Out = !trait.proj<@Dead[!T], "Out"> }
trait.trait private @Run(%self: !trait.claim<@Run[!T]>) { trait.assoc_type @Item }
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @Run_cond(%self: !trait.claim<@Run[!T]>,
                             %eq: !trait.claim<!trait.proj<@Has[!T], "Out"> = !trait.proj<@Other[!T], "Out">>) {
  trait.assoc_type @Item = i64
}

// ERASE: error: no impl with satisfiable assumptions for '!trait.proj<@Run[i32], "Item">'
// ERASE: error: equality '!trait.claim<!trait.proj<@Run[i32], "Item"> = i64>' does not settle before erasure: a side resolves to no type a predicate states
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@Run[i32], "Item">'}}
// expected-error @below {{equality '!trait.claim<!trait.proj<@Run[i32], "Item"> = i64>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func @assumed(%x: i64, %h: !trait.claim<!trait.proj<@Run[i32], "Item"> = i64>) -> i64 {
  return %x : i64
}

// -----

!T = !trait.poly<0>
trait.trait private @Dead(%self: !trait.claim<@Dead[!T]>) { trait.assoc_type @Out }
trait.trait private @Has(%self: !trait.claim<@Has[!T]>) { trait.assoc_type @Out }
trait.impl private @Has_any(%self: !trait.claim<@Has[!T]>) { trait.assoc_type @Out = !trait.proj<@Dead[!T], "Out"> }
trait.trait private @Other(%self: !trait.claim<@Other[!T]>) { trait.assoc_type @Out }
trait.impl private @Other_any(%self: !trait.claim<@Other[!T]>) { trait.assoc_type @Out = !trait.proj<@Dead[!T], "Out"> }
trait.trait private @Run(%self: !trait.claim<@Run[!T]>) { trait.assoc_type @Item }
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @Run_cond(%self: !trait.claim<@Run[!T]>,
                             %eq: !trait.claim<!trait.proj<@Has[!T], "Out"> = !trait.proj<@Other[!T], "Out">>) {
  trait.assoc_type @Item = i64
}

func.func private @need(!trait.claim<!trait.proj<@Run[i32], "Item"> = i64>)
func.func @alleged() {
  // ERASE: error: unproven monomorphic claim '!trait.claim<!trait.proj<@Run[i32], "Item"> = i64>' before erasure
  // expected-error @below {{'trait.allege' op no impl with satisfiable assumptions for '!trait.proj<@Run[i32], "Item">'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Run[i32], "Item"> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Run[i32], "Item"> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@Run[i32], "Item"> = i64>) -> ()
  return
}
