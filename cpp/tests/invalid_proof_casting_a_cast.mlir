// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A cast's body is the witness of a root -- a proof whose body holds its derive,
// or an unconditional impl -- coerced to the cast's spelling, so the impl, the
// arguments and the premises a cast records are one hop away. A witness in a
// proof's body naming a cast is refused where it is written: a proof casting
// itself, two proofs casting each other, and a cast of a cast all lead nowhere
// or the long way round to a root.

trait.trait private @M(%s: !trait.claim<@M[!trait.poly<0>]>) {}
trait.proof private @pa {
  // expected-error @below {{names the cast @pa: a witness in a proof's body names a root}}
  %w = trait.witness @pa for @M[i32]
  trait.return %w : !trait.claim<@M[i32] by @pa>
}

// -----

trait.trait private @M(%s: !trait.claim<@M[!trait.poly<0>]>) {}
trait.proof private @pa {
  %w = trait.witness @pb for @M[i32]
  trait.return %w : !trait.claim<@M[i32] by @pb>
}
trait.proof private @pb {
  // The verifier refuses the first witness of the cycle it reaches, which
  // rejects the module.
  // expected-error @below {{names the cast @pa: a witness in a proof's body names a root}}
  %w = trait.witness @pa for @M[i32]
  trait.return %w : !trait.claim<@M[i32] by @pa>
}

// -----

trait.trait private @M(%s: !trait.claim<@M[!trait.poly<0>]>) {}
trait.impl private @M_i32(%s: !trait.claim<@M[i32]>) {}
trait.proof private @pa {
  %w = trait.witness @M_i32 for @M[i32]
  trait.return %w : !trait.claim<@M[i32] by @M_i32>
}
trait.proof private @pb {
  // expected-error @below {{names the cast @pa: a witness in a proof's body names a root}}
  %w = trait.witness @pa for @M[i32]
  trait.return %w : !trait.claim<@M[i32] by @pa>
}
