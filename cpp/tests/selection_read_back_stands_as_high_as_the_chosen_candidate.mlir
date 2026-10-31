// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Selecting for @X[i32] judges @X_bad beside @X_good, and @X_bad's where
// clause opens a hundred-deep chain with no impl at its end. @X_good is
// chosen, and its own derivation is one frame deep, so the memo carries that
// height, not the refused candidate's: read back under @Z0's thirty-deep
// chain, the selection stands inside the depth limit and the program runs.

// CHECK: 6

!T = !trait.poly<0>
trait.trait private @X(!trait.claim<@X[!T]>) members [@X.m]
func.func private @X.m(!trait.claim<@X[!T]>) -> i64
trait.impl private @X_good(%s: !trait.claim<@X[i32]>) members [@X.m -> @X_good.m] {
}
func.func private @X_good.m(%s: !trait.claim<@X[i32]>) -> i64 {
  %c = arith.constant 5 : i64 return %c : i64
}
trait.impl private @X_bad(%s: !trait.claim<@X[!T]>, %n: !trait.claim<@Y0[!T]>) members [@X.m -> @X_bad.m] {
}
func.func private @X_bad.m(%s: !trait.claim<@X[!T]>, %n: !trait.claim<@Y0[!T]>) -> i64 {
  %c = arith.constant 6 : i64 return %c : i64
}
// REPEAT 0 99: trait.trait private @Y{k}(!trait.claim<@Y{k}[!T]>)
// REPEAT 0 98: trait.impl private @Y{k}_i(%s: !trait.claim<@Y{k}[i32]>, %n: !trait.claim<@Y{k+1}[i32]>) {}
trait.trait private @Z0(!trait.claim<@Z0[!T]>) members [@Z0.m]
func.func private @Z0.m(!trait.claim<@Z0[!T]>) -> i64
// REPEAT 1 29: trait.trait private @Z{k}(!trait.claim<@Z{k}[!T]>)
trait.impl private @Z0_i(%s: !trait.claim<@Z0[i32]>, %n: !trait.claim<@Z1[i32]>) members [@Z0.m -> @Z0_i.m] {
}
func.func private @Z0_i.m(%s: !trait.claim<@Z0[i32]>, %n: !trait.claim<@Z1[i32]>) -> i64 {
  %c = arith.constant 1 : i64 return %c : i64
}
// REPEAT 1 28: trait.impl private @Z{k}_i(%s: !trait.claim<@Z{k}[i32]>, %n: !trait.claim<@Z{k+1}[i32]>) {}
trait.impl private @Z29_i(%s: !trait.claim<@Z29[i32]>, %n: !trait.claim<@X[i32]>) {
}
func.func @main() -> i64 {
  %z = trait.allege @Z0[i32]
  %vz = trait.func.call @Z0.m(%z) : (!trait.claim<@Z0[i32]>) -> i64
  %x = trait.allege @X[i32]
  %vx = trait.func.call @X.m(%x) : (!trait.claim<@X[i32]>) -> i64
  %r = arith.addi %vx, %vz : i64
  return %r : i64
}
