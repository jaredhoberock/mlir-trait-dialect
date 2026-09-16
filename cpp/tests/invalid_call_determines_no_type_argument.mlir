// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The callee's V stands in one position only, the associated-type argument of a
// projection whose binding drops it: the declaration instantiated at V is i32
// whatever V is, so the call's types name no instance, and every V would satisfy
// the comparison. A call the reading cannot close is refused where it stands,
// naming the parameter, rather than lowering to an instance nothing determined.

!V = !trait.poly<11>

trait.trait private @Has[!trait.poly<1>] {
  trait.assoc_type @A<[!trait.poly<3>]>
}
trait.impl private @Has_i64 for @Has[i64] {
  trait.assoc_type @A<[!trait.poly<4>]> = i32
}

func.func private @g(%v: !trait.proj<@Has[i64], "A", [!V]>) -> i64 {
  %z = arith.constant 0 : i64
  return %z : i64
}

func.func @main(%b: i32) -> i64 {
  // expected-error @below {{call to @g determines no type argument for '!trait.poly<11>'}}
  %r = trait.func.call @g(%b) : (i32) -> i64
  return %r : i64
}
