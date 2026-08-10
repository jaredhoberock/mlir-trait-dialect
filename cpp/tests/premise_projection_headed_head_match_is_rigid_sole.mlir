// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The audit's head match instantiates ONLY the cited impl's own generics; the
// redex application stays rigid. The host impl's self argument is a projection
// spelling (proj<@Other[i64],"X">), so the premise redex application argument is
// that projection. Matching the cited impl's concrete head @Sib[i32] against
// @Sib[proj<@Other[i64],"X">] never resolves the projection through a
// module-visible impl of @Other, so the head does not reconcile and the impl is
// refused. Its byte-identical companion, ..._two, adds one unrelated impl of
// @Other and must reach the SAME verdict: an impl's birth cannot turn on the
// unrelated impls the module carries.

!S = !trait.poly<0>

trait.trait @Other[!S] {
  trait.assoc_type @X
}
trait.impl @Other_i64 for @Other[i64] {
  trait.assoc_type @X = i64
}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}
trait.impl @Sib_i32 for @Sib[i32] {
  trait.assoc_type @Elem = f32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{projection mismatch: expected '!trait.proj<@Other[i64], "X">' but found 'i32'}}
trait.impl @Host_p for @Host[!trait.proj<@Other[i64], "X">]
    premises [#trait<certificate !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> resolves f32 by @Sib_i32>] {
  func.func @make(%x: !trait.proj<@Other[i64], "X">) -> f32 {
    %r = ub.poison : f32
    return %r : f32
  }
}
