// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The trait declares @f over two type parameters of its own, so a caller
// supplies an argument for each. The impl's copy spells tuple<P> where the
// trait spells its parameter: the positional correspondence would accept it --
// the counts agree and the trait's signature instantiated at that spelling is
// the impl's -- but the copy stands only at the tuples, and a call at i64
// reaches no instance of it. An impl's copy renames the trait's parameters, it
// does not instantiate them.

trait.trait private @T[!trait.poly<0>] {
  func.func private @f(!trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>
}

// expected-error @below {{method 'f' spells 'tuple<!trait.poly<1>>' where trait '@T' declares the type parameter '!trait.poly<1>': an impl's copy of a method renames the trait's type parameters, one for one}}
trait.impl private @I for @T[i32] {
  func.func @f(%x: i32, %m: tuple<!trait.poly<1>>, %n: !trait.poly<2>) -> tuple<!trait.poly<1>> {
    return %m : tuple<!trait.poly<1>>
  }
}
