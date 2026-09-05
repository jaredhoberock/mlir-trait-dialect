// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A trait, an impl and a proof are templates: monomorphization cuts their
// instances and the collector after erase takes what nothing names. Collection
// may only take a symbol nothing outside its table may name, so each of the
// three is private from birth and a public one is refused where it is written
// rather than discovered at erase. Nothing here runs a pass: the refusal is the
// op's own verifier, so a row that forgets the keyword fails at parse.

// expected-error @below {{'trait.trait' op must not be public}}
trait.trait @Public[!trait.poly<0>] {
}

// -----

trait.trait private @T[!trait.poly<0>] {
}

// expected-error @below {{'trait.impl' op must not be public}}
trait.impl for @T[i32] {
}

// -----

trait.trait private @T[!trait.poly<0>] {
}

trait.impl private @T_i32 for @T[i32] {
}

// expected-error @below {{'trait.proof' op must not be public}}
trait.proof @T_p proves @T_i32 for @T[i32] given []
