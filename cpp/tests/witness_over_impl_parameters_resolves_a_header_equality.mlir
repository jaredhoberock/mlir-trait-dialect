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

trait.trait private @Bar[!S] {
  trait.assoc_type @Assoc
}

trait.impl private @Bar_box for @Bar[tuple<!S>] {
  trait.assoc_type @Assoc = !S
}

trait.trait private @Foo[!S]
    where [@Bar[!S], !trait.proj<@Bar[!S], "Assoc"> = !trait.proj<@Foo[!S], "U">] {
  trait.assoc_type @U
}

trait.impl private @Foo_box for @Foo[tuple<!S>]
    witnesses [#trait<witness !trait.proj<@Bar[tuple<!S>], "Assoc"> = !S by @Bar_box>] {
  trait.assoc_type @U = !S
}
