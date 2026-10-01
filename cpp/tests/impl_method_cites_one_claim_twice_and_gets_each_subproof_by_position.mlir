// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Tr_impl's where clause states @Mark[i32] twice, and @P discharges entry 0
// with @Mark_one (7) and entry 1 with @Mark_two (9). The two citations spell one
// claim, so only their positions say which subproof each means: cut out of the
// impl, the method gets the subproof @P names at each entry's position. One
// citation stands inside a nested region and the other in a second block, and
// the method answers 10 * entry 0 + entry 1, so a citation answered by the other
// entry's subproof, or both by one, changes the result.

// CHECK: {{^}}79{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { func.func private @value() -> i64 }
trait.trait private @Tr[!T] { func.func private @value() -> i64 }
trait.impl private @Mark_one for @Mark[i32] {
  func.func @value() -> i64 {
    %v = arith.constant 7 : i64
    return %v : i64
  }
}
trait.impl private @Mark_two for @Mark[i32] {
  func.func @value() -> i64 {
    %v = arith.constant 9 : i64
    return %v : i64
  }
}
trait.impl private @Tr_impl for @Tr[i32] where [@Mark[i32], @Mark[i32]] {
  func.func @value() -> i64 {
    %first = scf.execute_region -> i64 {
      %m0 = trait.assume 0 : !trait.claim<@Mark[i32]>
      %v0 = trait.method.call %m0 @Mark[i32]::@value() : () -> i64
      scf.yield %v0 : i64
    }
    cf.br ^second
  ^second:
    %m1 = trait.assume 1 : !trait.claim<@Mark[i32]>
    %v1 = trait.method.call %m1 @Mark[i32]::@value() : () -> i64
    %ten = arith.constant 10 : i64
    %scaled = arith.muli %first, %ten : i64
    %r = arith.addi %scaled, %v1 : i64
    return %r : i64
  }
}
trait.proof private @P proves @Tr_impl[] for @Tr[i32] given [@Mark_one, @Mark_two]
func.func @main() -> i64 {
  %p = trait.witness @P for @Tr[i32]
  %r = trait.method.call %p @Tr[i32]::@value() : () -> i64 by @P
  return %r : i64
}
