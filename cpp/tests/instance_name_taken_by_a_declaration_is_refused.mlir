// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -split-input-file -verify-diagnostics

// A declaration standing under the name the instance of `@id` at i64 takes is
// no instance: adopted, the call would name a symbol nothing defines. It is
// refused once, naming the call, which then survives the stage.

func.func private @id(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
// expected-error-re@+1 {{'func.func' op stands under @id_he63b698b22049d91, the name of the instance the call at loc({{.*}}:18:8) needs, which only a function this stage cuts takes}}
func.func private @id_he63b698b22049d91(i64) -> i64
func.func @main() -> i64 {
  %c = arith.constant 7 : i64
  // expected-error@+1 {{rewritable generic call survived instantiate-monomorphs}}
  %r = trait.func.call @id(%c) : (i64) -> i64
  return %r : i64
}

// -----

// An op of another kind standing under the instance's name would push the
// instance to another name; it is refused alike.

func.func private @id(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}
// expected-error-re@+1 {{'llvm.func' op stands under @id_he63b698b22049d91, the name of the instance the call at loc({{.*}}:37:8) needs, which only a function this stage cuts takes}}
llvm.func @id_he63b698b22049d91(%x: i64) -> i64 {
  llvm.return %x : i64
}
func.func @main() -> i64 {
  %c = arith.constant 7 : i64
  // expected-error@+1 {{rewritable generic call survived instantiate-monomorphs}}
  %r = trait.func.call @id(%c) : (i64) -> i64
  return %r : i64
}
