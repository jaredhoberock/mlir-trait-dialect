// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proof is ground: the stage writes one only at a monomorphic application,
// so a citation of it is one comparison. A polymorphic claim is proven by a
// derive in the template that holds its variables, and a proof over a type
// variable is refused where it is declared, however an impl covering every
// argument (impl<T> Trait<T> for Number) would serve it.

!T0 = !trait.poly<0>
!T1 = !trait.poly<1>

trait.trait private @Trait(%self: !trait.claim<@Trait[!T0, !T1]>) {
  trait.method @method(!T0, !T1) -> i64
}

!T2 = !trait.poly<2>
trait.impl private @Trait_impl(%self_claim: !trait.claim<@Trait[i64, !trait.poly<0>]>) {
  trait.method @method(%self: i64, %arg: !trait.poly<0>) -> i64 {
    trait.return %self : i64
  }
}

!T3 = !trait.poly<3>

// expected-error @below {{proves '!trait.claim<@Trait[i64, tuple<!trait.poly<3>>]>', which spells a type variable: a proof is ground}}
trait.proof private @Trait_proof {
  %d = trait.derive @Trait[i64, tuple<!T3>] from @Trait_impl given()
  trait.return %d : !trait.claim<@Trait[i64, tuple<!T3>]>
}
