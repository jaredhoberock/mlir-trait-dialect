// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A proof over a projection whose two impls bind each other's associated type
// spells an associated-type binding cycle: @Loop[i32]'s Output is @Loop[i64]'s
// Output and back. The projection has no normal form, and proof verification
// resolves it through the ground-projection lookup as it records the obligation.
// The nonconverging resolution is reported as a clean diagnostic on the proof
// and refuses verification -- it neither aborts the process nor admits the
// cyclic proof.

!T = !trait.poly<0>

trait.trait private @Loop[!T] {
  trait.assoc_type @Output
}

trait.impl private @Loop_i32 for @Loop[i32] {
  trait.assoc_type @Output = !trait.proj<@Loop[i64], "Output">
}

trait.impl private @Loop_i64 for @Loop[i64] {
  trait.assoc_type @Output = !trait.proj<@Loop[i32], "Output">
}

!W = !trait.poly<1>
trait.trait private @Wants[!W] {}

trait.impl private @Wants_impl for @Wants[!trait.proj<@Loop[i32], "Output">] {}

// expected-error @+1 {{projection normalization did not converge within 64 iterations}}
trait.proof private @p proves @Wants_impl for @Wants[!trait.proj<@Loop[i32], "Output">] given []
