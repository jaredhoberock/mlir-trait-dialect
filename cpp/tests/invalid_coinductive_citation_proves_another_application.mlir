// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proof citing itself stands for its own claim. @Foo_tuple at tuple<i32>
// leaves the obligation @Foo[i32], which nothing implements, so the
// self-citation discharges an application the proof does not prove. A
// coinductive citation is read by that same comparison, not by trait name.

trait.trait private @Foo[!trait.poly<0>] {}
trait.impl private @Foo_tuple for @Foo[tuple<!trait.poly<0>>] where [@Foo[!trait.poly<0>]] {}

// expected-error @below {{proof @p proves '!trait.claim<@Foo[tuple<i32>]>', which does not discharge the obligation '!trait.claim<@Foo[i32]>'}}
trait.proof private @p proves @Foo_tuple for @Foo[tuple<i32>] given [@p]
