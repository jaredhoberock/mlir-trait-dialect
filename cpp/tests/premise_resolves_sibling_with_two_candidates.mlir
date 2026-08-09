// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A host method returns a sibling trait's associated type. Substituting the
// host impl's self application mints the ground projection Sib[i64]::Elem, which
// this impl's own bindings do not resolve. TWO impls apply to @Sib[i64] -- a
// concrete one and a blanket one -- so module-visible impl lookup declines the
// redex (it resolves only a sole candidate) and cannot reduce the expected
// signature. A declared premise names WHICH impl resolves the redex, so the
// verifier audits that citation and replays it locally: the expected signature
// reduces to (i64) -> i32 and matches the impl method. The premise channel is
// strictly more expressive than the lookup, which refuses this two-candidate
// case.

!S = !trait.poly<0>

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

// Two candidates for @Sib[i64]; both bind Elem = i32.
trait.impl @Sib_i64 for @Sib[i64] {
  trait.assoc_type @Elem = i32
}
trait.impl @Sib_T for @Sib[!S] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// CHECK: trait.impl @Host_i64 for @Host[i64]premises [#trait<certificate!trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64>]
trait.impl @Host_i64 for @Host[i64]
    premises [#trait<certificate !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
