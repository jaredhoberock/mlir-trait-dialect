// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s

// An unproven claim parameter is valid in a polymorphic function. Its method
// call verifies using that parameter and the unused template is erased.

!T = !trait.poly<0>

trait.trait private @Unwrap[!T] {
  func.func private @unwrap(!T) -> !T
}

func.func private @caller(%claim: !trait.claim<@Unwrap[!T]>, %value: !T) -> !T {
  %result = trait.method.call %claim @Unwrap[!T]::@unwrap(%value) : (!T) -> !T
  return %result : !T
}

// CHECK: module {
// CHECK-NEXT: }
