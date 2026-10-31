// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(verify-acyclic-traits)' %s

// Tests that a trait can require its own associated type to satisfy itself.
//
// @Trait has the requirement @Trait[!trait.proj<@Trait[!S], "Assoc">] — a
// self-reference mediated by a projection type. This is not a real cycle
// because the projection resolves to a concrete type during monomorphization.

!S = !trait.poly<0>

trait.trait private @Trait(!trait.claim<@Trait[!S]>) -> !trait.claim<@Trait[!trait.proj<@Trait[!S], "Assoc">]> assoc {Assoc = 0} members [@Trait.get]
func.func private @Trait.get(!trait.claim<@Trait[!S]>, !S) -> !trait.proj<@Trait[!S], "Assoc">
