// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A region an op runs at run time declares no type variables of its own: it is
// interior to the body around it, and the clone's substitution rewrites what
// stands inside it. The reading that reports what the substitution could not
// replace descends there too, so a conditional whose results and yields carry
// the declaration's own variable is cloned, and one that reaches for a variable
// the call binds nothing for is refused at the declaration.

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

func.func private @in_scope(%c: i1, %x: !trait.poly<0>) -> !trait.poly<0> {
  %r = scf.if %c -> (!trait.poly<0>) {
    scf.yield %x : !trait.poly<0>
  } else {
    %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<0>
    scf.yield %a : !trait.poly<0>
  }
  return %r : !trait.poly<0>
}

func.func @host_in_scope(%c: i1, %v: i64) -> i64 {
  %r = trait.func.call @in_scope(%c, %v) {type_params = [!trait.poly<0>], type_args = [i64]} : (i1, i64) -> i64
  return %r : i64
}

// -----

// expected-error@+1 {{type variable '!trait.poly<7>' in the body of '@out_of_scope' is bound by no parameter of its declaration, so no instance can replace it}}
func.func private @out_of_scope(%c: i1, %x: !trait.poly<0>) -> !trait.poly<0> {
  %r = scf.if %c -> (!trait.poly<0>) {
    scf.yield %x : !trait.poly<0>
  } else {
    // expected-note@+1 {{mentioned here}}
    %a = builtin.unrealized_conversion_cast %x : !trait.poly<0> to !trait.poly<7>
    %b = builtin.unrealized_conversion_cast %a : !trait.poly<7> to !trait.poly<0>
    scf.yield %b : !trait.poly<0>
  }
  return %r : !trait.poly<0>
}

func.func @host_out_of_scope(%c: i1, %v: i64) -> i64 {
  // expected-error@+1 {{rewritable generic call survived instantiate-monomorphs}}
  %r = trait.func.call @out_of_scope(%c, %v) {type_params = [!trait.poly<0>], type_args = [i64]} : (i1, i64) -> i64
  return %r : i64
}
