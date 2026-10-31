// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A generic associated type has two lists of parameters standing over a
// binding, and a use supplies each from a different place: the impl header's
// parameters, labels 0 to n - 1, come from the application the impl is
// selected for, and the associated type's own, the labels after them, come
// from the projection's associated type arguments. The two lists cannot share
// a label, and a binding naming a label in neither has nothing to supply it,
// so the resolved type would carry a parameter no substitution reaches.

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

trait.trait private @Container(!trait.claim<@Container[!trait.poly<0>]>) assoc {Item = 1}

// expected-error@+1 {{binds associated type 'Item' to a type mentioning type parameter '!trait.poly<1>', which neither impl '@Container_i32' nor the associated type declares}}
trait.impl private @Container_i32(%self: !trait.claim<@Container[i32]>) bindings {Item = !trait.poly<1>} {
}

// -----

trait.trait private @Family(!trait.claim<@Family[!trait.poly<0>]>) assoc {A = 1}

// expected-error@+1 {{binds associated type 'A' to a type mentioning type parameter '!trait.poly<2>', which neither impl '@Family_blanket' nor the associated type declares}}
trait.impl private @Family_blanket(%self: !trait.claim<@Family[!trait.poly<0>]>) bindings {A = tuple<!trait.poly<0>, !trait.poly<1>, !trait.poly<2>>} {
}
