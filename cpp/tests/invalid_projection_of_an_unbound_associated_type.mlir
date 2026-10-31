// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @Gen declares and @Gen_i64 binds the associated type A alone. A projection
// of B names an associated type the trait does not declare, so nothing could
// ever bind it, and it is refused where it is spelled, even though the trait
// application itself has a unique implementation.

!T = !trait.poly<0>

trait.trait private @Gen(!trait.claim<@Gen[!T]>) assoc {A = 0}

trait.impl private @Gen_i64(%self: !trait.claim<@Gen[i64]>) bindings {A = i32} {
}

func.func @main() {
  // expected-error @below {{projection '!trait.proj<@Gen[i64], "B">' names no associated type of trait '"Gen"'}}
  %r = ub.poison : !trait.proj<@Gen[i64], "B">
  return
}
