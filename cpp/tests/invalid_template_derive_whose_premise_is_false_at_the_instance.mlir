// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @Vector_blanket applies where Tensor[T]::Shape is i64. Inside the template
// that premise reads over the template's own variable and is left to the
// instances; at the instance @Tensor_i8 binds Shape to tuple<i64, i64>, so the
// impl does not apply there. The derive names the impl it stands on, so the
// refusal names the premise rather than reporting only that the claim went
// unproven.

trait.trait private @Tensor[!trait.poly<0>] { trait.assoc_type @Shape }
trait.trait private @Vector[!trait.poly<0>] { func.func private @v() -> i64 }
trait.impl private @Vector_blanket for @Vector[!trait.poly<0>] where [@Tensor[!trait.poly<0>], !trait.proj<@Tensor[!trait.poly<0>], "Shape"> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @Tensor_i8 for @Tensor[i8] { trait.assoc_type @Shape = tuple<i64, i64> }
func.func private @f(%t: !trait.claim<@Tensor[!trait.poly<0>]>) -> i64 {
  // expected-error@+1 {{impl '@Vector_blanket' applies where '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' = 'i64', and nothing here makes 'tuple<i64, i64>' and 'i64' one type at '!trait.claim<@Vector[i8]>'}}
  %v = trait.derive @Vector[!trait.poly<0>] from @Vector_blanket given(%t) : (!trait.claim<@Tensor[!trait.poly<0>]>)
  %r = trait.method.call %v @Vector[!trait.poly<0>]::@v() : () -> i64
  return %r : i64
}
func.func @main() -> i64 {
  %t = trait.allege @Tensor[i8]
  %r = trait.func.call @f(%t) {type_params = [!trait.poly<0>], type_args = [i8]} : (!trait.claim<@Tensor[i8]>) -> i64
  return %r : i64
}
