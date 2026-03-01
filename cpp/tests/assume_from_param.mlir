// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Test that trait.assume can resolve claims from enclosing function parameters,
// not just from enclosing trait.trait or trait.impl regions.

!T = !trait.poly<0>

trait.trait @Foo[!T] {
  func.func private @foo(!T) -> !T
}

// CHECK-LABEL: func.func @standalone_poly
// CHECK: trait.assume @Foo[!trait.poly<0>]
func.func @standalone_poly(%c: !trait.claim<@Foo[!T]>, %x: !T) -> !T {
  %a = trait.assume @Foo[!T]
  %res = trait.method.call %a @Foo[!T]::@foo(%x)
    : (!T) -> !T
  return %res : !T
}
