// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A trait's declaration of a method says how many type parameters of its own a
// caller supplies. An impl's copy that binds fewer is a different declaration:
// a call naming an argument for the trait's parameter would have nothing in the
// copy to name. The correspondence between the two is positional, so equal
// counts are what it requires.

!S = !trait.poly<0>
!M = !trait.poly<1>

trait.trait private @Keep[!S] {
  func.func private @keep(!S, !M) -> !S
}

// expected-error @below {{method 'keep' binds 0 type parameter(s) of its own, but trait '"Keep"' declares it with 1}}
trait.impl private @Keep_i64 for @Keep[i64] {
  func.func @keep(%x: i64, %m: i32) -> i64 {
    return %x : i64
  }
}
