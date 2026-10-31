// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A call of a trait's member says nothing about the member's own type
// variables: the trait's arguments ride in the receiver claim, and V is read
// off the argument position it stands in. The verifier checks the actuals are
// the declaration instantiated at V := i32.
//
// RUN: mlir-opt %s | FileCheck %s

trait.trait private @Tr(!trait.claim<@Tr[!trait.poly<0>]>) members [@Tr.m]
func.func private @Tr.m(!trait.claim<@Tr[!trait.poly<0>]>, !trait.poly<0>, !trait.poly<1>) -> !trait.poly<1>

// CHECK: trait.func.call @Tr.m({{.*}}) : (!trait.claim<@Tr[i64]>, i64, i32) -> i32
func.func @caller(%x: i64, %y: i32) -> i32 {
  %c = trait.allege @Tr[i64]
  %r = trait.func.call @Tr.m(%c, %x, %y) : (!trait.claim<@Tr[i64]>, i64, i32) -> i32
  return %r : i32
}
