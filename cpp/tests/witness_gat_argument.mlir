// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A projection-resolution certificate carries the associated type's own type
// arguments (a generic associated type). The seam audit specializes the impl's
// binding for those arguments: @Trait[i64]::Assoc<i1> resolves to i1 because the
// impl binds Assoc<!poly<99>> = !poly<99>. A citation of the true contractum is
// accepted; a citation of a false one is refused.

trait.trait @Trait[!trait.poly<0>] {
  trait.assoc_type @Assoc<[!trait.poly<1>]>
}
trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Assoc<[!trait.poly<99>]> = !trait.poly<99>
}

func.func @true_citation() -> !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1> {
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Assoc", [i1]> resolves i1 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i1>
}

// -----

trait.trait @Trait[!trait.poly<0>] {
  trait.assoc_type @Assoc<[!trait.poly<1>]>
}
trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Assoc<[!trait.poly<99>]> = !trait.poly<99>
}

func.func @false_citation() -> !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i64> {
  // expected-error @below {{binds the redex to 'i1', not the certified contractum 'i64'}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Assoc", [i1]> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i64>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Assoc", [i1]> = i64>
}
