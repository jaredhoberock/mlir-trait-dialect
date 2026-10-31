// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Two impls of @Other bind @Other[i64]::X. A projection standing in the head of
// a proj_resolve witness's application is matched where it stands, so the
// citation of @Sib_i32 (an impl of @Sib[i32]) is refused for the projection
// @Sib[@Other[i64]::X]::Elem: a witness's verdict does not turn on resolving
// that head through the module's impls.

!S = !trait.poly<0>

trait.trait private @Other(!trait.claim<@Other[!S]>) assoc {X = 0}
trait.impl private @Other_i64(%self: !trait.claim<@Other[i64]>) bindings {X = i64} {
}
trait.impl private @Other_T(%self: !trait.claim<@Other[!S]>) bindings {X = i64} {
}

trait.trait private @Sib(!trait.claim<@Sib[!S]>) assoc {Elem = 0}
trait.impl private @Sib_i32(%self: !trait.claim<@Sib[i32]>) bindings {Elem = f32} {
}

trait.trait private @Host(!trait.claim<@Host[!S]>) members [@Host.make]
func.func private @Host.make(!trait.claim<@Host[!S]>, !S) -> !trait.proj<@Sib[!S], "Elem">

trait.impl private @Host_p(%self: !trait.claim<@Host[!trait.proj<@Other[i64], "X">]>) members [@Host.make -> @Host_p.make] {
}
func.func private @Host_p.make(%self: !trait.claim<@Host[!trait.proj<@Other[i64], "X">]>, %x: !trait.proj<@Other[i64], "X">) -> !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> {
  %r = ub.poison : f32
  // expected-error @below {{impl '@Sib_i32' at the arguments the citation gives it proves '!trait.claim<@Sib[i32]>', not '!trait.claim<@Sib[!trait.proj<@Other[i64], "X">]>'}}
  %e = trait.witness proj_resolve !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> resolves f32 by @Sib_i32
    : !trait.claim<!trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> = f32>
  %c = trait.coerce %r : f32 to !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> via (%e)
    : (!trait.claim<!trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem"> = f32>)
  return %c : !trait.proj<@Sib[!trait.proj<@Other[i64], "X">], "Elem">
}
