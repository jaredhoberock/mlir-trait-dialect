// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// An equality relates two types, and a claim is a value's evidence, never a
// type: an endpoint holding one, proven or not, at its root or inside another
// type, is refused where the equality is built, as an application refuses a
// claim argument. Otherwise a hypothesis equating two claims would let a coerce
// turn evidence of one application into evidence of another with no equality
// between their arguments, and one equating a claim with a value would turn
// evidence into a value.

// expected-error @below {{a type-equality endpoint must not contain a claim}}
func.func private @f(!trait.claim<!trait.claim<@Trait[i64] by @Trait_impl> = i64>)

// -----

// expected-error @below {{a type-equality endpoint must not contain a claim}}
func.func private @g(!trait.claim<!trait.claim<@T[i32]> = !trait.claim<@T[i64]>>)

// -----

// expected-error @below {{a type-equality endpoint must not contain a claim}}
func.func private @h(!trait.claim<tuple<!trait.claim<@T[i32]>> = tuple<!trait.claim<@T[i32]>>>)
