// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing -verify-diagnostics

// A trait whose `where` clause names a trait the module never defines. The
// monomorphize pass runs the acyclicity screen, which resolves each
// `where`-clause edge by name; the dangling reference is refused with a
// diagnostic rather than reaching the aborting trait accessor. The parser's
// verifier is disabled here so the screen faces the unverified IR a launch would
// hand it, which is the shape that formerly aborted the process.

!T = !trait.poly<0>

// expected-error @+1 {{trait `where` clause references undefined trait 'Undefined'}}
trait.trait @A[!T] where [@Undefined[!T]] {}
