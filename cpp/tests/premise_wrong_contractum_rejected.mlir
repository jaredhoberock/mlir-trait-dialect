// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A premise's certified contractum must be the type its cited impl actually
// binds. @Sib_i64 binds Sib[i64]::Elem to i32, but the premise certifies i64;
// the birth audit reads the impl's binding and refuses the mismatch, so a
// premise cannot smuggle a wrong resolution past the verifier.

!S = !trait.poly<0>

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64 for @Sib[i64] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{impl '@Sib_i64' binds the redex to 'i32', not the certified contractum 'i64'}}
trait.impl @Host_i64 for @Host[i64]
    premises [#trait<certificate !trait.proj<@Sib[i64], "Elem"> resolves i64 by @Sib_i64>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
