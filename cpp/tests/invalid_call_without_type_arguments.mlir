// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A generic callee's declaration binds parameters and a call supplies an
// argument for each of them. A call that names none says nothing about the
// instance it wants, and re-deriving it from spellings a sweep may have
// normalized is guessing, so it is refused where it stands -- for a free
// function's own parameters and for a method's alike.

func.func private @generic(%x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

func.func @calls_a_free_function(%x: i64) -> i64 {
  // expected-error @below {{call to @generic supplies 0 of its 1 type arguments}}
  %r = trait.func.call @generic(%x) : (i64) -> i64
  return %r : i64
}

// -----

!S = !trait.poly<0>
!M = !trait.poly<1>

trait.trait private @Keep[!S] {
  func.func private @keep(!S, !M) -> !S
}

trait.impl private @Keep_i64 for @Keep[i64] {
  func.func @keep(%x: i64, %m: !M) -> i64 {
    return %x : i64
  }
}

func.func @calls_a_method(%x: i64, %m: i32) -> i64 {
  %c = trait.allege @Keep[i64]
  // expected-error @below {{call to @keep supplies 0 of its 1 type arguments}}
  %r = trait.method.call %c @Keep[i64]::@keep(%x, %m) : (i64, i32) -> i64
  return %r : i64
}
