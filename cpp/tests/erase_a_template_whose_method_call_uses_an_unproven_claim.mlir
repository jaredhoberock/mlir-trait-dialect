// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s

// An unproven claim parameter is valid in a polymorphic function. Its member
// call verifies using that parameter and the unused template is erased.

!T = !trait.poly<0>

trait.trait private @Unwrap(!trait.claim<@Unwrap[!T]>) members [@Unwrap.unwrap]
func.func private @Unwrap.unwrap(!trait.claim<@Unwrap[!T]>, !T) -> !T

func.func private @caller(%claim: !trait.claim<@Unwrap[!T]>, %value: !T) -> !T {
  %result = trait.func.call @Unwrap.unwrap(%claim, %value) : (!trait.claim<@Unwrap[!T]>, !T) -> !T
  return %result : !T
}

// CHECK: module {
// CHECK-NEXT: }
