// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing -verify-diagnostics -split-input-file

// The acyclicity screen faces unverified IR ahead of the full verifier (a launch
// screening a frozen blob runs it before module verify), so it must refuse a
// malformed trait cleanly rather than dereference it. The parser's verifier is
// disabled on these rows so the screen sees the shapes a bytecode blob can carry
// but the text parser and full verifier reject.

// A trait op whose `where`-clause requirements property is absent: the screen
// refuses it rather than dereferencing the null array while iterating its edges.
// expected-error @+1 {{trait carries no `where`-clause requirements array}}
"trait.trait"() ({^bb0:}) {sym_name = "A"} : () -> ()

// -----

// A `where` clause naming a trait the module does not define: the screen resolves
// each edge by name and refuses the dangling reference rather than reaching the
// aborting trait accessor.
!T = !trait.poly<0>
// expected-error @+1 {{trait `where` clause references undefined trait 'Undefined'}}
trait.trait @A[!T] where [@Undefined[!T]] {}
