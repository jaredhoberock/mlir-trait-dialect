// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics -split-input-file

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}

trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) {}

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}

trait.impl private @B_impl(%self: !trait.claim<@B[!trait.poly<0>]>, %a: !trait.claim<@A[!trait.poly<0>]>) {}

// A proof's body cites a generic impl by name; an impl binding type parameters
// is cited only through a proof of its own.
trait.proof private @B_impl_p {
  // expected-error @below {{impl '@A_impl' binds type parameters or has a where clause, so it must be cited through a trait.proof}}
  %p0 = trait.witness @A_impl for @A[i8]
  %d = trait.derive @B[i8] from @B_impl given(%p0) : (!trait.claim<@A[i8] by @A_impl>)
  trait.return %d : !trait.claim<@B[i8]>
}
