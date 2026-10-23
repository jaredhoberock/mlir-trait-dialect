// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// A template's call standing inside a symbol table other than its module is
// never lowered where it stands while it is the template's
// (generic_call_inside_an_unused_template_in_another_symbol_table_is_collected.mlir),
// but @run instantiates @t, and the instance's clone of the call stands there
// outside every template, cut before any verification. Its lowering would be a
// func.call naming a function of that table alone, so it is not lowered, and
// instantiation names it by the rule that refuses such a call
// (invalid_trait_call_inside_another_symbol_table.mlir).

func.func private @f(%x: !trait.poly<0>) -> !trait.poly<0> { return %x : !trait.poly<0> }
func.func private @t(%x: !trait.poly<0>) -> !trait.poly<0> {
  "test.symbol_scope"() ({
    // expected-error @below {{'trait.func.call' op stands inside 'test.symbol_scope', a symbol table below the module its callee is named in}}
    // expected-error @below {{'trait.func.call' op rewritable generic call survived instantiate-monomorphs}}
    %r = trait.func.call @f(%x) : (!trait.poly<0>) -> !trait.poly<0>
    "test.finish"() : () -> ()
  }) : () -> ()
  return %x : !trait.poly<0>
}
func.func @run() -> i64 {
  %k = arith.constant 7 : i64
  %r = trait.func.call @t(%k) : (i64) -> i64
  return %r : i64
}
