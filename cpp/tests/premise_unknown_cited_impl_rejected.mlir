// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proj_resolve witness cites its resolving impl by symbol; verification looks
// it up in the module. A witness naming an impl that does not exist is refused,
// so it cannot cite a phantom resolver.

!S = !trait.poly<0>

trait.trait private @Sib(!trait.claim<@Sib[!S]>) assoc {Elem = 0}

trait.impl private @Sib_i64(%self: !trait.claim<@Sib[i64]>) bindings {Elem = i32} {
}

trait.trait private @Host(!trait.claim<@Host[!S]>) members [@Host.make]
func.func private @Host.make(!trait.claim<@Host[!S]>, !S) -> !trait.proj<@Sib[!S], "Elem">

trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) members [@Host.make -> @Host_i64.make] {
}
func.func private @Host_i64.make(%self: !trait.claim<@Host[i64]>, %x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
  %r = ub.poison : i32
  // expected-error @below {{cannot find trait.impl '@Nope' cited by the witness}}
  %e = trait.witness proj_resolve !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Nope
    : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i32>
  %c = trait.coerce %r : i32 to !trait.proj<@Sib[i64], "Elem"> via (%e)
    : (!trait.claim<!trait.proj<@Sib[i64], "Elem"> = i32>)
  return %c : !trait.proj<@Sib[i64], "Elem">
}
