// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A witness's seam audit rewrites its resolved binding by the cited equality
// premises. A premise whose left endpoint occurs in its right endpoint,
// !poly<0> = tuple<!poly<0>>, describes a rewrite with no finite fixed point.
// The audit refuses the premise rather than expanding the substitution forever,
// so the verifier stays total on spellable IR.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

func.func @f(%pre: !trait.claim<!S = tuple<!S>>) -> !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S> {
  // expected-error @below {{a self-referential equality premise ('!trait.poly<0>' occurs in its own rewrite) has no finite solution}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[!S], "Output"> resolves !S by @Trait_impl
    given(%pre) : (!trait.claim<!S = tuple<!S>>)
    : !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S>
  return %e : !trait.claim<!trait.proj<@Trait[!S], "Output"> = !S>
}
