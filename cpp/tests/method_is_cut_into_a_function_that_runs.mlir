// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=CUT
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// An impl's method is cut into a module-level func.func, every trait.return of
// its body becoming a func.return over the same operands: @pick returns from two
// blocks. A trait's default is cut from the trait itself the same way, its
// %self taking the receiver's proof. main computes
// 100 * pick(-5) + 10 * pick(5) + twice(4) = 100 * 1 + 10 * 2 + 2 * 2, so a
// return left standing or a return of the wrong block changes the result or
// refuses the module.

// CHECK: {{^}}124{{$}}

// CUT-LABEL: trait.trait private @Tr
// CUT:         trait.method @twice(
// CUT:           trait.return
// CUT:       func.func private @Tr_{{h[0-9a-f]+}}_twice(%{{.*}}: !trait.claim<@Tr[i64] by @Tr_i64>
// CUT-NOT:     trait.return
// CUT:         {{^ *}}return
// CUT-LABEL: trait.impl private @Tr_i64
// CUT:         trait.method @pick(
// CUT:           trait.return
// CUT:           trait.return
// CUT:       func.func private @Tr_i64_{{.*}}_pick(
// CUT-NOT:     trait.return
// CUT:         {{^ *}}return
// CUT-NOT:     trait.return
// CUT:         {{^ *}}return

!T = !trait.poly<0>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) {
  trait.method @pick(!T) -> i64
  trait.method @twice(%x: !T) -> i64 {
    %p = trait.method.call %self @Tr[!T]::@pick(%x) : (!T) -> i64
    %r = arith.addi %p, %p : i64
    trait.return %r : i64
  }
}

trait.impl private @Tr_i64(%self: !trait.claim<@Tr[i64]>) {
  trait.method @pick(%x: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %x, %zero : i64
    cf.cond_br %negative, ^below, ^above
  ^below:
    %one = arith.constant 1 : i64
    trait.return %one : i64
  ^above:
    %four = arith.constant 4 : i64
    %ge = arith.cmpi sge, %x, %four : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %r = arith.select %ge, %two, %three : i64
    trait.return %r : i64
  }
}

func.func @main() -> i64 {
  %w = trait.allege @Tr[i64]
  %minus5 = arith.constant -5 : i64
  %five = arith.constant 5 : i64
  %four = arith.constant 4 : i64
  %a = trait.method.call %w @Tr[i64]::@pick(%minus5) : (i64) -> i64
  %b = trait.method.call %w @Tr[i64]::@pick(%five) : (i64) -> i64
  %c = trait.method.call %w @Tr[i64]::@twice(%four) : (i64) -> i64
  %hundred = arith.constant 100 : i64
  %ten = arith.constant 10 : i64
  %a100 = arith.muli %a, %hundred : i64
  %b10 = arith.muli %b, %ten : i64
  %ab = arith.addi %a100, %b10 : i64
  %r = arith.addi %ab, %c : i64
  return %r : i64
}
