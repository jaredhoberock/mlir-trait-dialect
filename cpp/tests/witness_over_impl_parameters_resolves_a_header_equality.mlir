// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A generic impl of a trait carrying a header equality: @Foo's law relates the
// sibling projection proj<@Bar[!S],"Assoc"> to its own @U. The verifier
// enumerates no impls, so the law reduces only through the impl's own bindings
// and its declared witnesses. The witness here is spelled over the impl's own
// parameter and cites the BLANKET sibling @Bar_box, whose head the projection
// rigidly matches -- the evidence a generic impl's header equalities need,
// carrying no claim about any single instance.

// CHECK: trait.impl private @Foo_box

!S = !trait.poly<0>

trait.trait private @Bar(!trait.claim<@Bar[!S]>) assoc {Assoc = 0}

trait.impl private @Bar_box(%self: !trait.claim<@Bar[tuple<!S>]>) bindings {Assoc = !S} {
}

trait.trait private @Foo(!trait.claim<@Foo[!S]>) -> (!trait.claim<@Bar[!S]>, !trait.claim<!trait.proj<@Bar[!S], "Assoc"> = !trait.proj<@Foo[!S], "U">>) assoc {U = 0}

trait.impl private @Foo_box(%self: !trait.claim<@Foo[tuple<!S>]>) bindings {U = !S} {
  %req0 = trait.allege @Bar[tuple<!S>]
  %req1 = trait.allege !trait.proj<@Bar[tuple<!S>], "Assoc"> = !trait.proj<@Foo[tuple<!S>], "U">
  trait.return %req0, %req1 : !trait.claim<@Bar[tuple<!S>]>, !trait.claim<!trait.proj<@Bar[tuple<!S>], "Assoc"> = !trait.proj<@Foo[tuple<!S>], "U">>
}
