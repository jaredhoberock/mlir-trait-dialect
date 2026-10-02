// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proof citing itself stands for its own claim. @Foo_tuple at tuple<i32>
// takes a premise @Foo[i32], which nothing implements, so the self-citation
// in the proof's body supplies an application the proof does not prove. A
// coinductive citation is read by that same comparison, not by trait name.

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) {}
trait.impl private @Foo_tuple(%self: !trait.claim<@Foo[tuple<!trait.poly<0>>]>, %foo: !trait.claim<@Foo[!trait.poly<0>]>) {}

trait.proof private @p {
  // expected-error @below {{proof @p proves '!trait.claim<@Foo[tuple<i32>]>', which does not discharge the obligation '!trait.claim<@Foo[i32]>'}}
  %p0 = trait.witness @p for @Foo[i32]
  %d = trait.derive @Foo[tuple<i32>] from @Foo_tuple given(%p0) : (!trait.claim<@Foo[i32] by @p>)
  trait.return %d : !trait.claim<@Foo[tuple<i32>]>
}
