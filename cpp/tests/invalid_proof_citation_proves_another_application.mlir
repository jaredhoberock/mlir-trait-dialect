// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Requirement evidence stands for a requirement only when its claim is that
// requirement. @B[i32] requires @A[i32]; the impl returns @A_i64's evidence, an
// impl of the same trait at another argument. Reading the cited impl's own
// header against its own claim says nothing about the requirement, so the two
// must be compared.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {}

// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[i32]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}
