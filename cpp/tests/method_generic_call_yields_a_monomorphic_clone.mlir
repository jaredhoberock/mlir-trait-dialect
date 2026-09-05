// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A call to a method with its own type variable is cloned once, straight to the
// instance the call asks for. The impl's copy of the method spells that variable
// with its own name, so the call's binding is rekeyed through the correspondence
// between the trait method's variables and the impl method's before the clone is
// cut; a clone carrying the impl's bindings but not the call's would leave a
// module-level template between the call and its instance.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

!S = !trait.poly<0>
!V = !trait.poly<9>

trait.trait @Store[!S] {
  func.func private @keep(!S, !V) -> !V
}

trait.impl @Store_impl_i64 for @Store[i64] {
  func.func @keep(%self: i64, %v: !trait.poly<5>) -> !trait.poly<5> {
    return %v : !trait.poly<5>
  }
}

func.func @main(%x: i64, %v: i32) -> i32 {
  %p = trait.witness @Store_impl_i64 for @Store[i64]
  %r = trait.method.call %p @Store[i64]::@keep(%x, %v)
    : (i64, i32) -> i32
    by @Store_impl_i64
  return %r : i32
}

// No clone of the method is left polymorphic. A partly substituted clone prints
// ahead of the instance cut from it, so the exclusion leads: a CHECK-NOT scans
// only from the preceding match onward, and one placed after the positive check
// would never reach the line such a clone stands on.
// CHECK-NOT: func.func private @Store_impl_i64_keep_{{.*}}!trait.poly
// CHECK: func.func private @Store_impl_i64_keep_{{.*}}%arg2: i32) -> i32
// CHECK: func.func @main
// CHECK: call @Store_impl_i64_keep_
// CHECK-NOT: trait.method.call
