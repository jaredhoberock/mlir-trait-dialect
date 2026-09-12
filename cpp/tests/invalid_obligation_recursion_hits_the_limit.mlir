// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' 2>&1 | FileCheck %s

// The blanket impl proves @Foo[T] out of @Foo[tuple<T>], so discharging
// @Foo[i32] asks about @Foo[tuple<i32>], then @Foo[tuple<tuple<i32>>], and so
// on. Every step is a new application, so the same-application cycle guard
// never fires; the count of obligations naming one trait along the chain is
// what stops it, and the chain's ends say where the growth came from.
//
// The refusal stands at the demand. Every impl on the chain was asked about
// because something wanted the application in hand; naming one of them would
// name an impl with nothing wrong with it.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations of @Foo stand on the chain that reaches it
// CHECK-NEXT: trait.allege @Foo[i32]
// CHECK: note: required by {{.*}}@Foo[i32]
// CHECK: note: required by {{.*}}@Foo[tuple<i32>]
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @Foo[!trait.poly<0>] {
}

trait.impl private @Foo_blanket for @Foo[!trait.poly<1>] where [@Foo[tuple<!trait.poly<1>>]] {
}

func.func private @needs(%c: !trait.claim<@Foo[i32]>) {
  return
}

func.func @main() {
  %c = trait.allege @Foo[i32]
  func.call @needs(%c) : (!trait.claim<@Foo[i32]>) -> ()
  return
}
