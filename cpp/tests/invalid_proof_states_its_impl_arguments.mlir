// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The derived application fixes the impl's arguments: its header at them is
// the derived claim, and each premise is read at them.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A(%self: !trait.claim<@A[!S]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {}
trait.trait private @B(%self: !trait.claim<@B[!S]>) {}
trait.impl private @B_tuple(%self: !trait.claim<@B[tuple<!trait.poly<0>>]>, %a: !trait.claim<@A[!trait.poly<0>]>) {}
trait.proof private @p {
  %a = trait.witness @A_i32 for @A[i32]
  // expected-error @below {{premise 0 of impl '@B_tuple' is '!trait.claim<@A[i64]>', and the derive supplies '!trait.claim<@A[i32]>'}}
  %d = trait.derive @B[tuple<i64>] from @B_tuple given(%a) : (!trait.claim<@A[i32] by @A_i32>)
  trait.return %d : !trait.claim<@B[tuple<i64>]>
}

// -----

// One premise per where-clause entry.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @A(%self: !trait.claim<@A[!S]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {}
trait.trait private @C(%self: !trait.claim<@C[!S]>) { trait.assoc_type @Val }
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) { trait.assoc_type @Val = i64 }
trait.trait private @B(%self: !trait.claim<@B[!S]>) {}
trait.impl private @B_tuple(%self: !trait.claim<@B[tuple<!trait.poly<0>>]>, %a: !trait.claim<@A[!trait.poly<0>]>, %val: !trait.claim<!trait.proj<@C[!trait.poly<0>], "Val"> = i64>) {}
trait.proof private @p {
  %a = trait.witness @A_i32 for @A[i32]
  // expected-error @below {{impl '@B_tuple' has 2 where entries, and the citation supplies 1 claims}}
  %d = trait.derive @B[tuple<i32>] from @B_tuple given(%a) : (!trait.claim<@A[i32] by @A_i32>)
  trait.return %d : !trait.claim<@B[tuple<i32>]>
}

// -----

// An equality entry takes equality evidence, not the evidence of an impl.

!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @C(%self: !trait.claim<@C[!S]>) { trait.assoc_type @Val }
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) { trait.assoc_type @Val = i64 }
trait.trait private @B(%self: !trait.claim<@B[!S]>) {}
trait.impl private @B_tuple(%self: !trait.claim<@B[tuple<!trait.poly<0>>]>, %val: !trait.claim<!trait.proj<@C[!trait.poly<0>], "Val"> = i64>) {}
trait.proof private @p {
  %c = trait.witness @C_i32 for @C[i32]
  // expected-error @below {{premise 0 of impl '@B_tuple' is '!trait.claim<!trait.proj<@C[i32], "Val"> = i64>', and the derive supplies '!trait.claim<@C[i32]>'}}
  %d = trait.derive @B[tuple<i32>] from @B_tuple given(%c) : (!trait.claim<@C[i32] by @C_i32>)
  trait.return %d : !trait.claim<@B[tuple<i32>]>
}
