// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Byte-identical to ..._sole except for ONE added, unrelated impl of @Other.
// With the rigid head match the projection in the projection's application is never
// resolved through the module, so the added impl changes nothing: this impl
// reaches the SAME refusal its sole companion does. Before the head match was
// made rigid, the second impl made the inner projection two-candidate, the
// rebuild's lookup declined, the module-capable unifier tolerated the unresolved
// crossing, and the impl was accepted -- an unrelated impl flipping an impl-verification
// verdict. An impl's verdict no longer turns on unrelated module impls.

!S = !trait.poly<0>

trait.trait private @Other[!S] {
  trait.assoc_type @X
}
trait.impl private @Other_i64 for @Other[i64] {
  trait.assoc_type @X = i64
}
trait.impl private @Other_T for @Other[!S] {
  trait.assoc_type @X = i64
}

trait.trait private @Sib[!S] {
  trait.assoc_type @Elem
}
trait.impl private @Sib_i32 for @Sib[i32] {
  trait.assoc_type @Elem = f32
}

trait.trait private @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{type mismatch: expected '!trait.claim<@Sib[i32]>' but found '!trait.claim<@Sib[!trait.proj<@Other[i64], "X">]>'}}
trait.impl private @Host_p for @Host[!trait.proj<@Other[i64], "X">]
    witnesses [#trait<witness !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> = f32 by @Sib_i32>] {
  func.func @make(%x: !trait.proj<@Other[i64], "X">) -> f32 {
    %r = ub.poison : f32
    return %r : f32
  }
}
