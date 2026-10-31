// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Resolving the independent @Res[i64]::X projection cannot remove ambiguity
// between two @Gen[i64] impls. The @Gen[i64]::A projection remains an error:
// the projection is diagnosed where it survives, with the ambiguity selection
// refused it on.

!T = !trait.poly<0>

trait.trait private @Gen(!trait.claim<@Gen[!T]>) assoc {A = 0}

// expected-note@+1 {{candidate}}
trait.impl private @Gen_wide(%self: !trait.claim<@Gen[i64]>) bindings {A = i32} {
}

// expected-note@+1 {{candidate}}
trait.impl private @Gen_narrow(%self: !trait.claim<@Gen[i64]>) bindings {A = i16} {
}

trait.trait private @Res(!trait.claim<@Res[!T]>) assoc {X = 0}

trait.impl private @Res_i64(%self: !trait.claim<@Res[i64]>) bindings {X = i32} {
}

// expected-error @below {{incoherent impls (multiple satisfiable) for '!trait.proj<@Gen[i64], "A">'}}
// expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
func.func @main() -> (!trait.proj<@Gen[i64], "A">, !trait.proj<@Res[i64], "X">) {
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Gen[i64], "A">
  %s = ub.poison : !trait.proj<@Res[i64], "X">
  return %r, %s : !trait.proj<@Gen[i64], "A">, !trait.proj<@Res[i64], "X">
}
