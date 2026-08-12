// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A premise resolves only a GROUND sibling projection. Here the host impl is
// generic over !S and the premise projection proj<@Sib[!S],"Elem"> quantifies
// over that parameter while citing the single-instance impl @Sib_i64. Were the
// projection allowed, its poly variable would unify with the cited impl's
// concrete head and this generic impl would be accepted at birth on the strength
// of ONE instance. Verification refuses a non-ground projection outright,
// mirroring the guard the retired candidate lookup applied.

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

// expected-error @below {{premise projection '!trait.proj<@Sib[!trait.poly<0>], "Elem">' is not ground; a premise resolves only a ground sibling projection}}
trait.impl @Host_T for @Host[!S]
    premises [#trait<witness !trait.proj<@Sib[!S], "Elem"> = i32 by @Sib_i64>] {
  func.func @make(%x: !S) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
