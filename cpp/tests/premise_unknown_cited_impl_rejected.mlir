// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A premise cites its resolving impl by symbol; verification looks it up in the
// module. A premise naming an impl that does not exist is refused at birth, so
// a certificate cannot cite a phantom resolver.

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

// expected-error @below {{cannot find trait.impl '@Nope' cited by the certificate}}
trait.impl @Host_i64 for @Host[i64]
    witnesses [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i32 by @Nope>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
