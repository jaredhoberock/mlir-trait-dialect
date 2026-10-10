// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @Vector_blanket applies where Tensor[T]::Shape is i64. Inside the template
// the derive supplies that premise as an allegation over the template's own
// variable, left to the instances; at the instance @Tensor_i8 binds
// Shape to tuple<i64, i64>, so the allegation is refused where it stands, and
// it and the derive resting on it stand unproven at the instantiation check.

trait.trait private @Tensor(%self: !trait.claim<@Tensor[!trait.poly<0>]>) { trait.assoc_type @Shape }
trait.trait private @Vector(%self: !trait.claim<@Vector[!trait.poly<0>]>) { trait.method @v() -> i64 }
trait.impl private @Vector_blanket(%self: !trait.claim<@Vector[!trait.poly<0>]>, %tensor: !trait.claim<@Tensor[!trait.poly<0>]>, %shape: !trait.claim<!trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64>) {
  trait.method @v() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) { trait.assoc_type @Shape = tuple<i64, i64> }
func.func private @f(%t: !trait.claim<@Tensor[!trait.poly<0>]>) -> i64 {
  // expected-error@+2 {{alleges '!trait.proj<@Tensor[i8], "Shape">' = 'i64', and impl selection resolves its sides to 'tuple<i64, i64>' and 'i64'}}
  // expected-error@+1 {{unproven monomorphic claim '!trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64
  // expected-error@+1 {{unproven monomorphic claim '!trait.claim<@Vector[i8]>' after instantiate-monomorphs}}
  %v = trait.derive @Vector[!trait.poly<0>] from @Vector_blanket[!trait.poly<0>] given(%t, %e)
    : (!trait.claim<@Tensor[!trait.poly<0>]>, !trait.claim<!trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64>)
  %r = trait.method.call %v @Vector[!trait.poly<0>]::@v() : () -> i64
  return %r : i64
}
func.func @main() -> i64 {
  %t = trait.allege @Tensor[i8]
  %r = trait.func.call @f(%t) : (!trait.claim<@Tensor[i8]>) -> i64
  return %r : i64
}
