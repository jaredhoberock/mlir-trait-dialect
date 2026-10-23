// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @Outer_i32 returns @W9's witness for its requirement @Wrapped[i32], and
// @Wrapped_i32 returns @Nine's for @Mark[i32]. Projecting @Wrapped[i32] off a
// claim of @Outer[i32] proven by @O9 reads @W9 there, and projecting
// @Mark[i32] off that reads @Nine through @W9's impl, so the call runs @Nine's
// method though @Seven also serves @Mark[i32].

// CHECK-LABEL: func.func @main() -> i64
// CHECK:         call @Nine_{{h[0-9a-f]+}}_value()

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> {}
trait.trait private @Outer(%self: !trait.claim<@Outer[!T]>) -> !trait.claim<@Wrapped[!T]> {}
trait.impl private @Wrapped_i32(%self: !trait.claim<@Wrapped[i32]>) {
  %nine = trait.witness @Nine for @Mark[i32]
  trait.return %nine : !trait.claim<@Mark[i32] by @Nine>
}
trait.proof private @W9 {
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.impl private @Outer_i32(%self: !trait.claim<@Outer[i32]>) {
  %wrapped = trait.witness @W9 for @Wrapped[i32]
  trait.return %wrapped : !trait.claim<@Wrapped[i32] by @W9>
}
trait.proof private @O9 {
  %d = trait.derive @Outer[i32] from @Outer_i32 given()
  trait.return %d : !trait.claim<@Outer[i32]>
}
func.func @main() -> i64 {
  %o = trait.witness @O9 for @Outer[i32]
  %w = trait.project %o[0] : !trait.claim<@Outer[i32] by @O9> -> !trait.claim<@Wrapped[i32]>
  %m = trait.project %w[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
  %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
  return %v : i64
}
