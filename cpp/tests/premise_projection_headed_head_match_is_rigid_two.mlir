// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Byte-identical to ..._sole except for ONE added, unrelated impl of @Other.
// With the rigid head match the projection in the redex application is never
// resolved through the module, so the added impl changes nothing: this impl
// reaches the SAME refusal its sole companion does. Before the head match was
// made rigid, the second impl made the inner projection two-candidate, the
// rebuild's lookup declined, the module-capable unifier tolerated the unresolved
// crossing, and the impl was accepted -- an unrelated impl flipping a birth
// verdict. That estate dependence is gone.

!S = !trait.poly<0>

trait.trait @Other[!S] {
  trait.assoc_type @X
}
trait.impl @Other_i64 for @Other[i64] {
  trait.assoc_type @X = i64
}
trait.impl @Other_T for @Other[!S] {
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
    premises [#trait<witness !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> = f32 by @Sib_i32>] {
  func.func @make(%x: !trait.proj<@Other[i64], "X">) -> f32 {
    %r = ub.poison : f32
    return %r : f32
  }
}
